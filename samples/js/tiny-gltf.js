/**
 * TinyGltf
 * Loads glTF 2.0 file, resolves buffer and image dependencies, and computes node world transforms and aabbs.
 * This is a VERY simplified glTF loader that avoids doing too much work for you.
 * It should generally not be used outside of simple tutorials or examples.
 */

import { WebGPUMipmapGenerator } from './webgpu-mipmap-generator.js';
import { mat4, vec3 } from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.3/esm/index.js';

const GLB_MAGIC = 0x46546C67;
const CHUNK_TYPE = {
  JSON: 0x4E4F534A,
  BIN: 0x004E4942,
};

const DEFAULT_TRANSLATION = [0, 0, 0];
const DEFAULT_ROTATION = [0, 0, 0, 1];
const DEFAULT_SCALE = [1, 1, 1];

const absUriRegEx = new RegExp(`^${window.location.protocol}`, 'i');
const dataUriRegEx = /^data:/;
function resolveUri(uri, baseUrl) {
  if (!!uri.match(absUriRegEx) || !!uri.match(dataUriRegEx)) {
      return uri;
  }
  return baseUrl + uri;
}

// Very simple AABB tracking so that we can position cameras sensibly.
class AABB {
  min = vec3.fromValues(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
  max = vec3.fromValues(Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);

  constructor(aabb) {
    if(aabb) {
      vec3.copy(this.min, aabb.min);
      vec3.copy(this.max, aabb.max);
    }
  }

  union(other) {
    vec3.min(this.min, this.min, other.min);
    vec3.max(this.max, this.max, other.max);
  }

  transform(mat) {
    const corners = [
      [this.min[0], this.min[1], this.min[2]],
      [this.min[0], this.min[1], this.max[2]],
      [this.min[0], this.max[1], this.min[2]],
      [this.min[0], this.max[1], this.max[2]],
      [this.max[0], this.min[1], this.min[2]],
      [this.max[0], this.min[1], this.max[2]],
      [this.max[0], this.max[1], this.min[2]],
      [this.max[0], this.max[1], this.max[2]],
    ];

    vec3.set(this.min, Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
    vec3.set(this.max, Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);

    for (const corner of corners) {
      vec3.transformMat4(corner, corner, mat);
      vec3.min(this.min, this.min, corner);
      vec3.max(this.max, this.max, corner);
    }
  }

  get center() {
    return vec3.fromValues(
      ((this.max[0] + this.min[0]) * 0.5),
      ((this.max[1] + this.min[1]) * 0.5),
      ((this.max[2] + this.min[2]) * 0.5),
    );
  }

  get radius() {
    return vec3.distance(this.max, this.min) * 0.5;
  }
}

function setWorldMatrix(gltf, node, parentWorldMatrix) {
  // Don't recompute nodes we've already visited.
  if (node.worldMatrix) { return; }

  if (node.matrix) {
    node.worldMatrix = mat4.clone(node.matrix);
  } else {
    node.worldMatrix = mat4.create();
    mat4.fromRotationTranslationScale(
      node.worldMatrix,
      node.rotation,
      node.translation,
      node.scale);
  }

  mat4.multiply(node.worldMatrix, parentWorldMatrix, node.worldMatrix);

  // If the node has a mesh, get the AABB for that mesh and transform it to get the node's AABB.
  if ('mesh' in node) {
    const mesh = gltf.meshes[node.mesh];

    // Compute the mesh AABB if we haven't previously.
    if (!mesh.aabb) {
      mesh.aabb = new AABB();
      for (const primitive of mesh.primitives) {
        // The accessor has a min and max property
        mesh.aabb.union(gltf.accessors[primitive.attributes.POSITION]);
      }
    }

    node.aabb = new AABB(mesh.aabb);
    node.aabb.transform(node.worldMatrix);
  }

  if (node.children) {
    for (const childIndex of node.children) {
      const child = gltf.nodes[childIndex];
      setWorldMatrix(gltf, child, node.worldMatrix);

      if (child.aabb) {
        if (!node.aabb) {
          node.aabb = new AABB(child.aabb);
        } else {
          node.aabb.union(child.aabb)
        }
      }
    }
  }
}

export class TinyGltf {
  loadImageSlots = undefined;

  async loadFromUrl(url) {
    const i = url.lastIndexOf('/');
    const baseUrl = (i !== 0) ? url.substring(0, i + 1) : '';
    const response = await fetch(url);

    if (url.endsWith('.gltf')) {
      return this.loadFromJson(await response.json(), baseUrl);
    } else if (url.endsWith('.glb')) {
      return this.loadFromBinary(await response.arrayBuffer(), baseUrl);
    } else {
      throw new Error('Unrecognized file extension');
    }
  }

  async loadFromBinary(arrayBuffer, baseUrl) {
    const headerView = new DataView(arrayBuffer, 0, 12);
    const magic = headerView.getUint32(0, true);
    const version = headerView.getUint32(4, true);
    const length = headerView.getUint32(8, true);

    if (magic != GLB_MAGIC) {
      throw new Error('Invalid magic string in binary header.');
    }

    if (version != 2) {
      throw new Error('Incompatible version in binary header.');
    }

    let chunks = {};
    let chunkOffset = 12;
    while (chunkOffset < length) {
      const chunkHeaderView = new DataView(arrayBuffer, chunkOffset, 8);
      const chunkLength = chunkHeaderView.getUint32(0, true);
      const chunkType = chunkHeaderView.getUint32(4, true);
      chunks[chunkType] = arrayBuffer.slice(chunkOffset + 8, chunkOffset + 8 + chunkLength);
      chunkOffset += chunkLength + 8;
    }

    if (!chunks[CHUNK_TYPE.JSON]) {
      throw new Error('File contained no json chunk.');
    }

    const decoder = new TextDecoder('utf-8');
    const jsonString = decoder.decode(chunks[CHUNK_TYPE.JSON]);
    return this.loadFromJson(JSON.parse(jsonString), baseUrl, chunks[CHUNK_TYPE.BIN]);
  }

  async loadFromJson(json, baseUrl, binaryChunk = null) {
    if (!baseUrl) {
      throw new Error('baseUrl must be specified.');
    }

    if (!json.asset) {
      throw new Error('Missing asset description.');
    }

    if (json.asset.minVersion != '2.0' && json.asset.version != '2.0') {
      throw new Error('Incompatible asset version.');
    }

    // Resolve defaults for as many properties as we can.
    for (const accessor of json.accessors) {
      accessor.byteOffset = accessor.byteOffset ?? 0;
      accessor.normalized = accessor.normalized ?? false;
    }

    for (const bufferView of json.bufferViews) {
      bufferView.byteOffset = bufferView.byteOffset ?? 0;
    }

    for (const node of json.nodes) {
      if (!node.matrix) {
        node.rotation = node.rotation ?? DEFAULT_ROTATION;
        node.scale = node.scale ?? DEFAULT_SCALE;
        node.translation = node.translation ?? DEFAULT_TRANSLATION;
      }
    }

    if (json.samplers) {
      for (const sampler of json.samplers) {
        sampler.wrapS = sampler.wrapS ?? GL.REPEAT;
        sampler.wrapT = sampler.wrapT ?? GL.REPEAT;
      }
    }

    // Resolve buffers and images first, since these are the only external resources that the file
    // might reference.
    // Buffers will be exposed as ArrayBuffers.
    // Images will be exposed as ImageBitmaps.

    // Buffers
    const pendingBuffers = [];
    if (binaryChunk) {
      pendingBuffers.push(Promise.resolve(binaryChunk));
    } else {
      for (const index in json.buffers) {
        const buffer = json.buffers[index];
        const uri = resolveUri(buffer.uri, baseUrl);
        pendingBuffers[index] = fetch(uri).then(response => response.arrayBuffer());
      }
    }

    // If the loader has been instructed to only load certain images (such as just baseColorTexture)
    // then scan through all the materials first and gather the image IDs for only those images.
    // (This feature really only makes sense for this specific set of samples.)
    let activeImageSet;
    if (this.loadImageSlots) {
      activeImageSet = new Set();
      for (const material of json.materials) {
        for (const imageChannel of this.loadImageSlots) {
          const texture = material[imageChannel] ?? material.pbrMetallicRoughness[imageChannel];
          if (texture !== undefined) {
            activeImageSet.add(json.textures[texture.index].source);
          }
        }
      }
    }

    // Images
    const pendingImages = [];
    for (let index = 0; index < json.images?.length || 0; ++index) {
      if (activeImageSet && !activeImageSet.has(index)) { continue; }
      const image = json.images[index];
      if (image.uri) {
        pendingImages[index] = fetch(resolveUri(image.uri, baseUrl)).then(async (response) => {
          return createImageBitmap(await response.blob());
        });
      } else {
        const bufferView = json.bufferViews[image.bufferView];
        pendingImages[index] = pendingBuffers[bufferView.buffer].then((buffer) => {
          const blob = new Blob(
            [new Uint8Array(buffer, bufferView.byteOffset, bufferView.byteLength)],
            {type: image.mimeType});
          return createImageBitmap(blob);
        });
      }
    }

    // Compute a world transform for each node, starting at the root nodes and
    // working our way down.
    for (const scene of Object.values(json.scenes)) {
      for (const nodeIndex of scene.nodes) {
        const node = json.nodes[nodeIndex];
        setWorldMatrix(json, node, mat4.create());

        if (node.aabb) {
          if (!scene.aabb) {
            scene.aabb = new AABB(node.aabb);
          } else {
            scene.aabb.union(node.aabb)
          }
        }
      }
    }

    // Replace the resolved resources in the JSON structure.
    json.buffers = await Promise.all(pendingBuffers);
    json.images = await Promise.all(pendingImages);

    return json;
  }

  static componentCountForType(type) {
    switch (type) {
      case 'SCALAR': return 1;
      case 'VEC2': return 2;
      case 'VEC3': return 3;
      case 'VEC4': return 4;
      default: return 0;
    }
  }

  static sizeForComponentType(componentType) {
    switch (componentType) {
      case GL.BYTE: return 1;
      case GL.UNSIGNED_BYTE: return 1;
      case GL.SHORT: return 2;
      case GL.UNSIGNED_SHORT: return 2;
      case GL.UNSIGNED_INT: return 4;
      case GL.FLOAT: return 4;
      default: return 0;
    }
  }

  static packedArrayStrideForAccessor(accessor) {
    return TinyGltf.sizeForComponentType(accessor.componentType) * TinyGltf.componentCountForType(accessor.type);
  }
}

/**
 * TinyGltfWebGPU
 * Loads glTF 2.0 file and creates the necessary WebGPU buffers, textures, and samplers for you.
 * As with the base TinyGltf, this is a VERY simplified loader and should not be used outside of
 * simple tutorials or examples.
 */

// To make it easier to reference the WebGL enums that glTF uses.
const GL = WebGLRenderingContext;

function gpuAddressModeForWrap(wrap) {
  switch (wrap) {
    case GL.CLAMP_TO_EDGE: return 'clamp-to-edge';
    case GL.MIRRORED_REPEAT: return 'mirror-repeat';
    default: return 'repeat';
  }
}

function createGpuBufferFromBufferView(device, bufferView, buffer, usage) {
  // For our purposes we're only worried about bufferViews that have a vertex or index usage.
  if (!usage) { return null; }

  const gpuBuffer = device.createBuffer({
    label: bufferView.name,
    // Round the buffer size up to the nearest multiple of 4.
    size: Math.ceil(bufferView.byteLength / 4) * 4,
    usage: usage,
    mappedAtCreation: true,
  });

  const gpuBufferArray = new Uint8Array(gpuBuffer.getMappedRange());
  gpuBufferArray.set(new Uint8Array(buffer, bufferView.byteOffset, bufferView.byteLength));
  gpuBuffer.unmap();

  return gpuBuffer;
}

function createGpuSamplerFromSampler(device, sampler = {name: 'glTF default sampler'}) {
  const descriptor = {
    label: sampler.name,
    addressModeU: gpuAddressModeForWrap(sampler.wrapS),
    addressModeV: gpuAddressModeForWrap(sampler.wrapT),
  };

  if (!sampler.magFilter || sampler.magFilter == GL.LINEAR) {
    descriptor.magFilter = 'linear';
  }

  switch (sampler.minFilter) {
    case GL.NEAREST:
      break;
    case GL.LINEAR:
    case GL.LINEAR_MIPMAP_NEAREST:
      descriptor.minFilter = 'linear';
      break;
    case GL.NEAREST_MIPMAP_LINEAR:
      descriptor.mipmapFilter = 'linear';
      break;
    case GL.LINEAR_MIPMAP_LINEAR:
    default:
      descriptor.minFilter = 'linear';
      descriptor.mipmapFilter = 'linear';
      break;
  }

  return device.createSampler(descriptor);
}

function createGpuTextureFromImage(device, source, mipmapGenerator) {
  const mipLevelCount = Math.floor(Math.log2(Math.max(source.width, source.height))) + 1;
  const descriptor = {
    size: {width: source.width, height: source.height},
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    mipLevelCount
  };

  const texture = device.createTexture(descriptor);
  device.queue.copyExternalImageToTexture({source}, {texture}, descriptor.size);
  mipmapGenerator.generateMipmap(texture, descriptor);

  return texture;
}

export class TinyGltfWebGpu extends TinyGltf {
  constructor(device) {
    super();

    this.device = device;
    this.mipmapGenerator = new WebGPUMipmapGenerator(device);
    this.defaultSampler = createGpuSamplerFromSampler(device);
  }

  async loadFromJson(json, baseUrl, binaryChunk) {
    // Load the glTF file
    const gltf = await super.loadFromJson(json, baseUrl, binaryChunk);

    // Create the WebGPU resources
    const device = this.device;

    // Identify all the vertex and index buffers by iterating through all the primitives accessors
    // and marking the buffer views as vertex or index usage.
    // (There's technically a target attribute on the buffer view that's supposed to tell us what
    // it's used for, but that appears to be rarely populated.)
    const bufferViewUsages = [];
    function markAccessorUsage(accessorIndex, usage) {
      const accessor = gltf.accessors[accessorIndex];
      bufferViewUsages[accessor.bufferView] |= usage;
    }
    for (const mesh of gltf.meshes) {
      for (const primitive of mesh.primitives) {
        if ('indices' in primitive) {
          markAccessorUsage(primitive.indices, GPUBufferUsage.INDEX);
        }
        for (const attribute of Object.values(primitive.attributes)) {
          markAccessorUsage(attribute, GPUBufferUsage.VERTEX);
        }
      }
    }

    // Create WebGPU objects for all necessary buffers, images, and samplers
    gltf.gpuBuffers = [];
    for (const [index, bufferView] of Object.entries(gltf.bufferViews)) {
      gltf.gpuBuffers[index] = createGpuBufferFromBufferView(device, bufferView, gltf.buffers[bufferView.buffer], bufferViewUsages[index]);
    }

    gltf.gpuTextures = [];
    if (gltf.images?.length) {
      const imageTextures = [];
      if (gltf.images) {
        for (const [index, image] of Object.entries(gltf.images)) {
          if (!image) { continue; }
          imageTextures[index] = createGpuTextureFromImage(device, image, this.mipmapGenerator);
        }
      }

      const gpuSamplers = [];
      if (gltf.samplers) {
        for (const [index, sampler] of Object.entries(gltf.samplers)) {
          gpuSamplers[index] = createGpuSamplerFromSampler(device, sampler);
        }
      }

      if (gltf.textures) {
        for (const [index, texture] of Object.entries(gltf.textures)) {
          const imageTexture = imageTextures[texture.source];
          if (!imageTexture) { continue; }
          gltf.gpuTextures[index] = {
            texture: imageTexture,
            sampler: texture.sampler ? gpuSamplers[texture.sampler] : this.defaultSampler,
          }
        }
      }
    }
    gltf.gpuDefaultSampler = this.defaultSampler;

    return gltf;
  }

  static gpuFormatForAccessor(accessor) {
    const norm = accessor.normalized ? 'norm' : 'int';
    const count = TinyGltf.componentCountForType(accessor.type);
    const x = count > 1 ? `x${count}` : '';
    switch (accessor.componentType) {
      case GL.BYTE: return `s${norm}8${x}`;
      case GL.UNSIGNED_BYTE: return `u${norm}8${x}`;
      case GL.SHORT: return `s${norm}16${x}`;
      case GL.UNSIGNED_SHORT: return `u${norm}16${x}`;
      case GL.UNSIGNED_INT: return `u${norm}32${x}`;
      case GL.FLOAT: return `float32${x}`;
    }
  }

  static gpuPrimitiveTopologyForMode(mode) {
    switch (mode) {
      case GL.TRIANGLES: return 'triangle-list';
      case GL.TRIANGLE_STRIP: return 'triangle-strip';
      case GL.LINES: return 'line-list';
      case GL.LINE_STRIP: return 'line-strip';
      case GL.POINTS: return 'point-list';
    }
  }

  static gpuIndexFormatForComponentType(componentType) {
    switch (componentType) {
      case GL.UNSIGNED_SHORT: return 'uint16';
      case GL.UNSIGNED_INT: return 'uint32';
      default: return 0;
    }
  }
}
