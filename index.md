---
layout: page
title: Efficently rendering glTF models
subtitle: A WebGPU Case Study
menubar_toc: true
---

## Introduction

The WebGPU API is designed to make communicating between your code (ie: JavaScript in a browser) and the GPU more efficient. It builds on top of modern, fast native APIs such as Vulkan, Metal, or Direct3D 12, and as a result surfaces many of the same patterns for interacting with the GPU that those APIs established. One of the primary patterns that are seen in these APIs to achive this efficiency is defining the resources and state needed for rendering early on in large atomic bundles which can be validated once and creation time. This allows those same resources and state to be set quickly when it comes time to render with minimal overhead.

It's a highly effective pattern, but one that can initially feel awkward to work with for developers coming from APIs that allowed state to be defined more piecemeal at render time. As a result, new WebGPU developers may run into tricky situations when starting a new project or porting older code to the newer API and find themselves applying patterns that don't feel like an effective use of the API simply because it's difficult to see how to handle it better.

This document will walk through one such common and potentially confusing scenario: rendering [glTF 2.0 models](https://www.khronos.org/gltf/). Given that the design of glTF was heavily influenced by the WebGL/OpenGL ES APIs, it offers a good case-study for ways to approach these types of problems in WebGPU that will hopefully be applicable to situations that extend well beyond use of the format itself.

Having said that, it's worth keeping in mind that any patterns used to improve rendering efficiency will have upsides and downsides. There's rarely a "perfect" solution for any given problem, only a solution that works well with the tradeoffs your app is willing to make. Nothing in this doc should be seen as the definitive Correct Way To Do Things™️, it's just a collection of ways you _can_ apply WebGPU features to this particular use case.

## Who this document is for

Despite the fact that we'll be covering ways of rendering glTF models with WebGPU this is _not_ a glTF or WebGPU tutorial. Most of the glTF loading and parsing will be handwaved as "and then a library handles this part", and we're not going to spend any time talking about WebGPU basics like initialization, core API usage, or the shader language.

Instead we'll be focusing on how the data contained in the glTF maps to various WebGPU concepts, why some of those mappings can initially lead to inefficent use of WebGPU, and strategies for improving it.

Ideally you'd want to read through this document after you've _at least_ done a few "Hello world" WebGPU excercises where you've been able to get triangles on the screen, and it would be helpful if you take some time to look over the [glTF 2.0 reference guide](https://www.khronos.org/files/gltf20-reference-guide.pdf) if you're not already familiar with the file format.

If you haven't done any WebGPU development before, these are some great resources to start with!

 - [WebGPU Samples](https://austin-eng.com/webgpu-samples/)
 - [Raw WebGPU](https://alain.xyz/blog/raw-webgpu) - A WebGPU rendering overview/tutorial
 - [WebGPU — All of the cores, none of the canvas](https://surma.dev/things/webgpu/) - A WebGPU introduction focused on compute
 - The [WebGPU](https://gpuweb.github.io/gpuweb/) and [WGSL](https://gpuweb.github.io/gpuweb/wgsl/) specs - Dense and not fun to read, but a good reference

## Part 1: A Naive Renderer

Let's start by looking at what it takes to do the most straighforward "get triangles on the screen" renderer we can for a glTF model.

### A brief primer on glTF meshes

glTF is a popular format for delivering and loading runtime 3D assets, especially because it was designed to work well with web-friendly concepts like JSON and ArrayBuffers. As mentioned earlier, it was also designed with any eye towards being easy to display with WebGL (Or OpenGL ES), using enum values from that API to encode certain peices of state. It's absolutely possible to load glTF assets and display them efficiently with WebGPU, but this slight bias towards the older API means that the data as represented in the file sometimes needs to be transformed to fit the expected structure.

A perfect example of this is how glTF encodes vertex buffer data. glTF files define `meshes` which each contain a list of chunks of renderable geometry called (confusingly) `primitives`. A `primitive` is represented as a list of named `attributes`, and a `mode` (what WebGPU calls "primitive topology"). Primitives also reference a `material` and possibly `indices`, but let's ignore those for the moment to make things simpler.

```json
"meshes": [{
  "primitives": [{
    "mode": 4, // gl.TRIANGLES
    "attributes": {
      "POSITION": 0,
      "TEXCOORD_0": 1
    },
  }]
}]
```

(Aside: Personally I think `primitives` is a terrible name for this, as a "primitive" suggests to me a single point, line, or triangle, not a whole list of them. I would have gone with something like "submesh". But since `primitives` is the term used by glTF I'll use it throughout this doc to refer to the same concept for consistency.)

The attributes are indexes into an array of what glTF calls "Accessors", which in turn point into an array of "Buffer Views" that describes a range of a larger buffer and how the data is laid out within it:

```json
"accessors": [{          // POSITION attribute
  "bufferView": 0,
  "byteOffset": 0,
  "type": "VEC3",
  "componentType": 5126, // gl.FLOAT
  "normalized": false,
  "count": 8,
}, {                     // TEXCOORD_0 attribute
  "bufferView": 1,
  "byteOffset": 0,
  "type": "VEC2",
  "componentType": 5126, // gl.FLOAT
  "normalized": false,
  "count": 8,
}],

"bufferViews": [{
  "buffer": 0,
  "byteOffset": 0,
  "byteLength": 96,
  "byteStride": 12,
}, {
  "buffer": 0,
  "byteOffset": 96,
  "byteLength": 64,
  "byteStride": 8,
}]
```

Finally, you need to know where each mesh should be displayed in the scene, and that's defined by a tree of `nodes`, each of which can have a transform and, optionally, define which `mesh` should be rendered at that transform. (A single `mesh` could be referenced by multiple `nodes`, making each `node` effectively an instance of that `mesh`.)

```json
"nodes": [{
  "translation": [ 1,0,0 ],
  "mesh": 0
}, {
  "translation": [ -1,0,0 ],
  "mesh": 0
}]
```

### Rendering with WebGL

In WebGL, you would generally expect to upload the data from the Buffer View pointed at by each primitive attribute as a vertex buffer. Then at render time you'd, walk through the `node` tree and for each `primitive` of each `mesh` make a `gl.vertexAttribPointer()` call for each `attribute` with the data from their `accessor`, almost directly:

```js
// Simplified WebGL glTF rendering
function drawGLTFMesh(gltf, node) {
  gl.useProgram(shaderProgram);
  gl.uniformMatrix4fv(modelMatrixLocation, false, getWorldTransformForNode(node));

  const mesh = gltf.meshes[node.mesh];
  for (const primitive of mesh.primitives) {
    let drawCount;
    for (const [attribName, accessorIndex] of Object.entries(primitive.attributes)) {
      const attribLocation = gl.getAttribLocation(shaderProgram, attribName);
      const accessor = gltf.accessors[accessorIndex];
      const bufferView = gltf.bufferViews[accessor.bufferView];

      gl.bindBuffer(gl.ARRAY_BUFFER, glBufferForBufferView(bufferView));
      gl.enableVertexAttribArray(attribLocation);
      gl.vertexAttribPointer(
        attribLocation, numberOfComponentsForType(accessor.type), accessor.componentType,
        accessor.normalized, bufferView.byteStride, accessor.byteOffset);

      drawCount = accessor.count; // All attributes should have the same count.
    }

    gl.drawArrays(primitive.mode, 0, drawCount);
  }
}
```

That code snippet is far from optimal, but the point is to show how the glTF data structure maps to WebGL geometry. And as you can see it's relatively straightforward!

<details markdown=block>
  <summary markdown=span><b>Click here if you want to get pedantic about WebGL</b></summary>
  Okay, yes. The code above has a lot of issues but that's not the point of this article! Since you've dug into this section, though, here's some things the above snippet should be doing instead:

   - Attribute locations should be looked up ahead of time and cached, definitely not queried every time we draw!
   - Better yet, attribute locations should be set as shader program creation time with `gl.bindAttribLocation()` rather than looked up with
   - The code should be using Vertex Array Objects (VAOs), either from WebGL2 or the `OES_vertex_array_object` extension, to define the buffer bindings and vertex attrib pointers once at load time, which drastically reduces the number of calls in the render loop.
   - There's no go reason to keep the original glTF structures around for rendering. The values that are needed for the draw loop, like the primitive mode, draw count, and VAOs should be cached in a form that's easier to iterate through.
   - If all the meshes are using the same shader program it should be set outside of this function.
   - If you're using WebGL 2 you should be using uniform buffer objects (UBOs) instead of calling `uniformMatrix4fv()`.
   - I sure hope that `getWorldTransformForNode()` method isn't recalculating the matrix from scratch every frame!
   - Yes, materials, camera uniforms, etc, are being completely ignored here. How long do you want this doc to be?!?
   - And finally many of the tips below about sorting buffers and reducing state changes ALSO apply to WebGL!
</details><br/>

### A naive first pass at WebGPU rendering

So how would we render the same data with WebGPU?

### Vertex buffer uploading

As with the WebGL version, you would first identify all the `bufferViews` referenced by the `primitive.attributes` and upload their data to `GPUVertexBuffer`s. This is relatively straightforward and we won't spend much time on it here, the trickiest bit is simply remembering that WebGPU buffer sizes must be a multiple of 4, a restriction that didn't exist in WebGL. As such we need to round up the allocated size to the nearest multiple of four. It also makes it easier in this case to use buffer mapping rather than `writeBuffer()` to set the data, since `writeBuffer()` also requires that the data size to upload be a multiple of 4 but `TypedArray.set()` can handle any size.

```js
function createVertexBufferForBufferView(bufferView) {
  const buffer = getArrayBufferForGltfBuffer(bufferView.buffer);

  const gpuBuffer = device.createBuffer({
    // Round the buffer size up to the nearest multiple of 4.
    size: Math.ceil(bufferView.byteLength / 4) * 4,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const gpuBufferArray = new Uint8Array(gpuBuffer.getMappedRange());
  gpuBufferArray.set(new Uint8Array(buffer, bufferView.byteOffset, bufferView.byteLength));
  gpuBuffer.unmap();

  return gpuBuffer;
}
```

### Primitive pipelines

The next thing that you'll notice if you start digging into it is that the buffer/attribute properties that we spend so much of the WebGL rendering code setting up aren't defined during the render loop at all in WebGPU. Instead, they're set as part of a much larger bundle of state called the `GPURenderPipeline`.

A `GPURenderPipeline` contains the majority of the state needed for rendering, such as the shaders, vertex layout, culling behavior, blending behavior, etc. The few bits of rendering state that aren't part of the pipeline are things like viewport and scissor rects. If you are familiar with an API like WebGL, thinking about all this state as a single monolithic object can be difficult, because you're likely used to being able to configure them all separately with calls like `gl.enable()` and `gl.vertexAttribPointer()`.

Render pipelines are also fairly expensive to create, and can cause hitches if you are creating them while rendering. As a result we'll want to build all of our render pipelines at load time. And since glTF doesn't offer any guarantees about the structure or order of it's attribute data, which is part of the render pipeline state, we'll start by creating a pipeline that matches the requirements for each `primitive` in file.

```js
// A naive first pass at defining glTF geometry layout for WebGPU

// We can map the attributes to any location index we want as long as we're consistent
// between the pipeline definitions and the shader source.
const ShaderLocations = {
  POSITION: 0,
  NORMAL: 1,
};

// This will be used to store WebGPU information about our glTF primitives.
primitiveGpuData = new Map();

function setupPrimitive(gltf, primitive) {
  const bufferLayout = [];
  const gpuBuffers = [];
  let drawCount = 0;

  // Loop through every attribute in the primitive and build a description of the vertex
  // layout, which is needed to create the render pipeline.
  for (const [attribName, accessorIndex] of Object.entries(primitive.attributes)) {
    const accessor = gltf.accessors[accessorIndex];
    const bufferView = gltf.bufferViews[accessor.bufferView];

    // Get the shader location for this attribute. If it doesn't have one skip over the
    // attribute because we don't need it for rendering (yet).
    const shaderLocation = ShaderLocations[attribName];
    if (shaderLocation === undefined) { continue; }

    // Create a new vertex buffer entry for the render pipeline that describes this
    // attribute. Implicitly assumes that one buffer will be bound per attribute, even if
    // the attribute data is interleaved.
    bufferLayout.push({
      arrayStride: bufferView.byteStride,
      attributes: [{
        shaderLocation,
        format: gpuFormatForAccessor(accessor),
        offset: accessor.byteOffset,
      }]
    });

    // Since we're skipping some attributes, we need to track the WebGPU buffers that are
    // used here so that we can bind them in the correct order at draw time.
    gpuBuffers.push(gpuBufferForBufferView(accessor.bufferView));

    // All attributes should have the same count, which will be the draw count for
    // non-indexed geometry.
    drawCount = accessor.count;
  }

  // Create a render pipeline that is compatible with the vertex buffer layout for this primitive.
  const module = getShaderModule();
  const pipeline = device.createRenderPipeline({
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: bufferLayout,
    },
    primitive: {
      topology: gpuPrimitiveTopologyForMode(primitive.mode),
    },
    // ...Other properties omitted to avoid clutter.
  });

  // Store data needed to render this primitive.
  primitiveGpuData.set(primitive, { pipeline, buffers: gpuBuffers, drawCount });
}
```

I'm going to glaze over the shader returned by `getShaderModule()` because it's not particularly important at this point. All we care about is getting the geometry on screen, so the shader can simply consume the vertex attributes, apply the appropriate transform, and output white triangles. (I gave it some really simple lighting in the live samples so you could see the shape of the geometry better.)

<details markdown=block>
  <summary markdown=span><b>Click here if you want to see the shader code anyway</b></summary>

```js
function getShaderModule() {
  // Cache the shader module, since all the pipelines use the same one.
  if (!shaderModule) {
    // The shader source used here is intentionally minimal. It just displays the geometry
    // as white with a very simplistic directional lighting based only on vertex normals
    // (just to show the shape of the mesh a bit better.)
    const code = `
      // These are being managed in the demo base code.
      struct Camera {
        projection : mat4x4<f32>,
        view : mat4x4<f32>,
      };
      @group(0) @binding(0) var<uniform> camera : Camera;

      // This comes from the bind groups being created in setupMeshNode in the next section.
      @group(1) @binding(0) var<uniform> model : mat4x4<f32>;

      // These locations correspond with the values in the ShaderLocations struct in our JS and, by
      // extension, the buffer attributes in the pipeline vertex state.
      struct VertexInput {
        @location(${ShaderLocations.POSITION}) position : vec3<f32>,
        @location(${ShaderLocations.NORMAL}) normal : vec3<f32>,
      };

      struct VertexOutput {
        // Always need to at least output something to the position builtin.
        @builtin(position) position : vec4<f32>,

        // The other locations can be anything you want, as long as it's consitent between the
        // vertex and fragment shaders. Since we're defining both in the same module and using the
        // same structure for the input and output, we get that alignment for free!
        @location(0) normal : vec3<f32>,
      };

      @vertex
      fn vertexMain(input : VertexInput) -> VertexOutput {
        // Determines the values that will be sent to the fragment shader.
        var output : VertexOutput;

        // Transform the vertex position by the model/view/projection matrices.
        output.position = camera.projection * camera.view * model * vec4(input.position, 1.0);

        // Transform the normal by the model and view matrices. Normally you'd just do model matrix,
        // but adding the view matrix in this case is a hack to always keep the normals pointing
        // towards the light, so that we can clearly see the geometry even as we rotate it.
        output.normal = normalize((camera.view * model * vec4(input.normal, 0.0)).xyz);

        return output;
      }

      // Some hardcoded lighting constants.
      const lightDir = vec3(0.25, 0.5, 1.0);
      const lightColor = vec3(1.0, 1.0, 1.0);
      const ambientColor = vec3(0.1, 0.1, 0.1);

      @fragment
      fn fragmentMain(input : VertexOutput) -> @location(0) vec4<f32> {
        // An extremely simple directional lighting model, just to give our model some shape.
        let N = normalize(input.normal);
        let L = normalize(lightDir);
        let NDotL = max(dot(N, L), 0.0);

        // Surface color will just be the light color, so everything will appear white/grey.
        let surfaceColor = ambientColor + NDotL;

        // No transparency at this point.
        return vec4(surfaceColor, 1.0);
      }
    `;

    shaderModule = device.createShaderModule({ code });
  }

  return shaderModule;
}
```
</details>


### Transform bind groups

Next, in order for each of the meshes to be rendered in the correct place, we also need to have a uniform that contains the transform matrix for the node that the mesh is attached to. Fortunately this is less complicated.

```js
// This will be used to store WebGPU information about our nodes.
const nodeGpuData = new Map();

// Only called for nodes that have a 'mesh' property.
function setupMeshNode(gltf, node) {
  // Create a uniform buffer for this node and populate it with the node's world transform.
  const nodeUniformBuffer = device.createBuffer({
    size: 16 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(nodeUniformBuffer, 0, getWorldTransformForNode(gltf, node));

  // Create a bind group containing the uniform buffer for this node.
  const bindGroup = device.createBindGroup({
    layout: nodeBindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: nodeUniformBuffer },
    }],
  });

  nodeGpuData.set(node, { bindGroup });
}
```

### Render loop

Once you've created all the necessary pipelines and bind groups, you can then begin drawing the glTF model in your render loop, which is probably going to look something like this:

```js
function renderGltf(gltf, renderPass) {
  // Sets uniforms for things like that don't change for the entire frame,
  // like the projection and view matrices.
  renderPass.setBindGroup(0, frameBindGroup);

  // Loop through all of the nodes that we created transform uniforms for and set those bind groups.
  for (const [node, gpuNode] of nodeGpuData) {
    renderPass.setBindGroup(1, gpuNode.bindGroup);

    // Find the mesh for this node and loop through all of its primitives.
    const mesh = gltf.meshes[node.mesh];
    for (const primitive of mesh.primitives) {
      const gpuPrimitive = primitiveGpuData.get(primitive);

      // Set the pipeline for this primitive.
      renderPass.setPipeline(gpuPrimitive.pipeline);

      // Set the vertex buffers for this primitive.
      for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
        renderPass.setVertexBuffer(bufferIndex, gpuBuffer);
      }

      // Draw!
      renderPass.draw(gpuPrimitive.drawCount);
    }
  }
}
```

And that will get geometry on the screen! It's not a terrible place to start when you just want to see something working. (Some simple things have been omitted from the code above like handling glTF defaults, camera uniforms, and indexed geometry handling, but the fundamentals are the same.)

If you want to see the above approach in action, I've put together a sample page that does exactly that. It hides most of the boilerplate of initializing WebGPU, loading the glTF files, etc. and just focuses on the above methods, but you can see that using it we can load and render a variety of glTF models as simple untextured geometry. Even "larger" scenes like Sponza work!

[![Sample 1 sceenshot](/assets/images/sample-01.jpg)
Click to launch **Sample 01 - Naive Rendering**](01-naive-renderer.html)

So... triangles on screen! Victory! Slap some materials on there and call it a day, right?

Well, there's some unintuitive edge cases that you may not run into right away but could cause this rendering code to fail it you did encounter it.

Also, this approach will probably be fine for individual models on at least modestly powerful devices. But what if you aspire to bigger things? You want to perform well on the most lowely mobile devices, or be able to render much bigger scenes comprised of many models! This naive approach probably won't hold up.

So, what are some techniques that we can use to improve on this first pass?

## Part 2: Improving buffer bindings.

### Handling large attribute offsets

First we should fix a correctness issue that we'll face when loading some models. On the sample page linked above you may have noticed that if you try to load the "buggy" model, you get an error. On Chrome it'll read something like this:

```
Attribute offset (41448) with format VertexFormat::Float32x3 (size: 12) doesn't fit in the maximum vertex buffer stride (2048).
 - While validating attributes[0].
 - While validating buffers[1].
 - While validating vertex state.
 - While calling [Device].CreateRenderPipeline([RenderPipelineDescriptor "glTF renderer pipeline"]).
```

If you're new to WebGPU that kind of error can be a little intimidating, but it's telling us exactly what we need to know. We tried to create a pipeline where the 1st attribute (`attributes[0]`) of the second buffer (`buffers[1]`) had an offset (41448 bytes) that was invalid.

This is because in the code snippet above we're feeding the `accessor.byteOffset` directly into the buffer attribute `offset` when defining the render pipeline. This works well if the attributes begin near the beginning of the buffer view, but WebGPU will reject them if they larger than the `arrayStride` specified for the buffer, and the `arrayStride` can't be larger than than 2048 bytes.

For example, assume you have the following accessors:

```json
"accessors": [{          // "POSITION" attribute
  "bufferView": 0,
  "byteOffset": 0,
  "type": "VEC3",
  "componentType": 5126, // gl.FLOAT
  "normalized": false,
  "count": 480,
}, {                     // "NORMAL" attribute
  "bufferView": 0,
  "byteOffset": 5760,
  "type": "VEC3",
  "componentType": 5126, // gl.FLOAT
  "normalized": false,
  "count": 480,
}],

"bufferViews": [{
  "buffer": 0,
  "byteOffset": 0,
  "byteLength": 11520,
  "byteStride": 12,
}]
```

You can see that both of these point at the same `bufferView`, but while the first has a `byteOffset` of 0 the second points to a location 5760 byte in! This is because whatever tool produced this file made the decision to place all the attributes in a single buffer, one after the other like this:

```
Position|Position|Position|...|Normal|Normal|Normal|...
```

That's a perfectly valid thing to do! And were we writing a WebGL renderer it wouldn't cause us any problems, because `gl.vertexAttribPointer()` doesn't place any limits on the attribute offsets.

WebGPU, however, has two places where byte offsets into a vertex buffer can be specified: When setting the attribute offsets in a render pipeline, which is limited to the aforementioned `arrayStride`/2048 bytes, and as an optional argument when calling `setVertexBuffer()`, which has no limit. So, the intended way for you to specify an attribute offset is to "normalize" all the attribute offsets for a given buffer so that they the lowest attribute offset is treated as 0, then specify the offset to that attribute when binding the vertex buffer.

We can implement this for our code above like so:

```js
function setupPrimitive(gltf, primitive) {
  const bufferLayout = [];
  const gpuBuffers = [];
  let drawCount = 0;

  for (const [attribName, accessorIndex] of Object.entries(primitive.attributes)) {
    const accessor = gltf.accessors[accessorIndex];
    const bufferView = gltf.bufferViews[accessor.bufferView];

    const shaderLocation = ShaderLocations[attribName];
    if (shaderLocation === undefined) { continue; }

    bufferLayout.push({
      arrayStride: bufferView.byteStride,
      attributes: [{
        shaderLocation,
        format: gpuFormatForAccessor(accessor),
        offset: 0,
      }]
    });

    gpuBuffers.push({
      buffer: gpuBufferForBufferView(accessor.bufferView),
      // Save the attribute offset as a buffer offset instead.
      offset: accessor.byteOffset
    });

    drawCount = accessor.count;
  }

  primitiveGpuData.set(primitive, {
    // Moved the pipeline creation to a helper function to help keep these code
    // snippets focused.
    pipeline: getPipelineForPrimitive(gltf, primitive, bufferLayout.values()),
    buffers: gpuBuffers,
    drawCount
  });
}
```

And, as mentioned above, now that we're tracking buffer offsets along with the buffers we want to bind, we have to make sure that we apply those in the render loop:

```js
function renderGltf(gltf, renderPass) {
  renderPass.setBindGroup(0, frameBindGroup);

  for (const [node, gpuNode] of nodeGpuData) {
    renderPass.setBindGroup(1, gpuNode.bindGroup);

    const mesh = gltf.meshes[node.mesh];
    for (const primitive of mesh.primitives) {
      const gpuPrimitive = primitiveGpuData.get(primitive);
      renderPass.setPipeline(gpuPrimitive.pipeline);

      for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
        // Only change to the render loop is that we start setting offsets for the
        // vertex buffers now.
        renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
      }

      renderPass.draw(gpuPrimitive.drawCount);
    }
  }
}
```

With that modification some models that previously were previously failing to render, such as the "buggy" model in the sample above, start showing up! Hooray!

### Reduced binding for interleaved buffers

There's even more than we can improve regarding our buffer binding, though! One of the simplest things that we can address is the fact that if any of the the vertex attributes share a buffer the above code will still end up binding that single buffer to multiple slots, which is unnecessary overhead. Consider the following partial glTF file:

```json
"accessors": [{          // "POSITION" attribute
  "bufferView": 0,
  "byteOffset": 0,
  "type": "VEC3",
  "componentType": 5126, // gl.FLOAT
  "normalized": false,
  "count": 8,
}, {                     // "TEXCOORD_0" attribute
  "bufferView": 0,
  "byteOffset": 12,
  "type": "VEC2",
  "componentType": 5126, // gl.FLOAT
  "normalized": false,
  "count": 8,
}],

"bufferViews": [{
  "buffer": 0,
  "byteOffset": 0,
  "byteLength": 160,
  "byteStride": 20,
}]
```

The two `accessors` that both point at a different `byteOffset` into the same `bufferView`, like the previous example, but this time the offsets between them fit within the `byteStride`. This means that the vertex data in the buffer is laid out like so:

```
Position|TexCoord|Position|TexCoord|Position|TexCoord...
```

This is referred to as "interleaved" vertex data, and just like with the previous section it's a valid choice that the tools which produced the glTF file can make about how to lay out their data.

To take advantage of files with this kind of layout and reduce the number of times we need to call `setVertexBuffer()`, we can sort the attributes by the `bufferView` they use as we build the buffer layout. The one thing that we need to be careful of is that if we run into the situation from the previous step, where attributes share a buffer but aren't actually interleaved, we still need to treat those as separate buffers. It adds some complexity, but it's managable:

```js
function setupPrimitive(gltf, primitive) {
  // Note that these are maps now!
  const bufferLayout = new Map();
  const gpuBuffers = new Map();
  let drawCount = 0;

  for (const [attribName, accessorIndex] of Object.entries(primitive.attributes)) {
    const accessor = this.gltf.accessors[accessorIndex];
    const bufferView = this.gltf.bufferViews[accessor.bufferView];

    const shaderLocation = ShaderLocations[attribName];
    if (shaderLocation === undefined) { continue; }

    let buffer = bufferLayout.get(accessor.bufferView);
    let gpuBuffer;
    // If the delta between attributes falls outside the bufferView's stated arrayStride,
    // then the buffers should be considered separate.
    let separate = buffer && (Math.abs(accessor.byteOffset - buffer.attributes[0].offset) >= buffer.arrayStride);
    // If we haven't seen this buffer before OR have decided that is should be separate because its
    // offset is too large, create a new buffer entry for the pipeline's vertex layout.
    if (!buffer || separate) {
      buffer = {
        arrayStride: bufferView.byteStride,
        attributes: [],
      };
      // If the buffers are separate due to offset, don't use the bufferView index to track them.
      // Use the attribName instead, which is guaranteed to be unique.
      bufferLayout.set(separate ? attribName : accessor.bufferView, buffer);
      // We're going to start tracking the gpuBuffers by the buffer layout now rather than
      // the bufferView, since we might end up with multiple buffer layouts all
      // pointing at the same bufferView.
      gpuBuffers.set(buffer, {
        buffer: this.gltf.gpuBuffers[accessor.bufferView],
        offset: accessor.bufferView
      });
    } else {
      gpuBuffer = gpuBuffers.get(buffer);
      // Track the minimum offset across all attributes that share a buffer.
      gpuBuffer.offset = Math.min(gpuBuffer.offset, accessor.bufferView);
    }

    // Add the attribute to the buffer layout
    buffer.attributes.push({
      shaderLocation,
      format: gpuFormatForAccessor(accessor),
      offset: accessor.bufferView,
    });

    drawCount = accessor.count;
  }

  // For each buffer, normalize the attribute offsets by subtracting the buffer offset from
  // the attribute offsets.
  for (const buffer of bufferLayout.values()) {
    const gpuBuffer = gpuBuffers.get(buffer);
    for (const attribute of buffer.attributes) {
      attribute.offset -= gpuBuffer.offset;
    }
  }

  primitiveGpuData.set(primitive, {
    pipeline: getPipelineForPrimitive(gltf, primitive, bufferLayout.values()),
    buffers: gpuBuffers,
    drawCount
  });
}
```

And fortunately for us, this change doesn't require any changes to the render loop. We already did everything necessary in the last step.

<details markdown=block>
  <summary markdown=span><b>Click here to get pedantic about buffer grouping</b></summary>
  There's an edge case that the above code isn't handling well. Specifically, there's a risk that if a file is mixing both interleaved vertex data AND non-interleaved vertex data that shares a buffer then we may end up not properly identifying the interleaved data depending on order we process the attributes in, and bind the vertex buffers more times than is strictly necessary.

  In practice, though, this isn't really an issue. Tools which produce glTF files tend to stick with a single vertex layout pattern for the entire file, so while you may end up seeing non-interleaved shader buffers in one file and interleaved shared buffers in another, it's unlikely that you'll get both in a single file. And if you do it's probably a rare enough edge case that you don't need to spend much time trying to optimize it. The above code will still allow it to render correctly regardless.
</details>

You can see the combined changes from these two steps at work on the second sample page, which now loads every model in the list correctly.

[![Sample 2 sceenshot](/assets/images/sample-02.jpg)
Click to launch **Sample 02 - Buffer Layouts**](02-buffer-layouts.html)

### More work at load == faster drawing

You may be picking up on a pattern at this point that the setup code is getting more complex, but in return our drawing code can do less work. In this case that comes from looping over fewer vertex buffers, because we're doing the necessary grouping at setup time. This a good pattern, and it reflects the ethos of the WebGPU API as well: Do as much work as possible up front to make the most critical loop, drawing, faster. It's what the majority of this document is focused on.

So now we've reduced the amount of times we need to call `setVertexBuffer()` to a minimum, which is great! But ultimately that's a pretty minor performance concern compared to the elephant in the room...

### There's too many pipelines!

It's likely that even with a cursory look at the code above you can start to guess at one of the biggest efficiency issues it faces: _It creates a new pipeline for every single primitive_. That means that if your scene is comprised of 500 glTF `primitives`, even if they all share the same vertex layout, you will still end up with 500 `GPURenderPipelines` to switch between.

In WebGPU, one of the most important things you can do to improve the efficiency of your rendering is to minimize the number of `GPURenderPipeline` objects you need to switch between. Calls to `setPipeline()` should be treated as expensive, because they generally are! The more times you need to switch between pipelines in the course of rendering your scene, the more state that needs to be pushed to the GPU, and the less you can render overall.

With that in mind, let's examine ways we can reduce the amount of pipeline switching that happens in our code.

### Render Pipeline Structure

To start, let's take a closer look at what a `GPURenderPipeline` contains, and how it affects our rendering.

You can see what's in the pipeline by looking at the [WebGPU spec's `GPURenderPipelineDescriptor` definition](https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpipelinedescriptor) but given that it's a heavily nested structure it takes a bit of navigation to see the full thing.

<details markdown=block>
  <summary markdown=span><b>Click here to see an example of the full GPURenderPipelineDescriptor structure</b></summary>

```js
{
  layout: pipelineLayout,
  vertex: {
    module: gpuShaderModule,
    entryPoint: "vertexMain",
    buffers: [{
      arrayStride: 16,
      stepMode: "vertex",
      attributes: [{
        format: "float32x4",
        offset: 0,
        shaderLocation: 0,
      }],
    }],
    constants: {
      constantName: 1.0,
    },
  },
  primitive: {
    topology: "triangle-strip",
    stripIndexFormat: "uint32",
    frontFace: "ccw",
    cullMode: "none",
  },
  depthStencil: {
    format: "depth24plus-stencil8",
    depthWriteEnabled: true,
    depthCompare: "less",
    stencilFront: {
      compare: "always",
      failOp: "keep",
      depthFailOp: "keep",
      passOp: "keep",
    },
    stencilBack:{
      compare: "always",
      failOp: "keep",
      depthFailOp: "keep",
      passOp: "keep",
    },
    stencilReadMask: 0xFFFFFFFF,
    stencilWriteMask: 0xFFFFFFFF,
    depthBias: 0,
    depthBiasSlopeScale: 0,
    depthBiasClamp: 0,
  },
  multisample: {
    count: 4,
    mask: 0xFFFFFFFF,
    alphaToCoverageEnabled: true,
  },
  fragment: {
    module: gpuShaderModule,
    entryPoint: "fragmentMain",
    targets: [{
      format: "bgra8unorm",
      blend: {
        color: {
          operation: "add",
          srcFactor: "one",
          dstFactor: "zero",
        },
        alpha: {
          operation: "add",
          srcFactor: "one",
          dstFactor: "zero",
        },
      },
      writeMask: GPUColorWrite.ALL,
    }],
    constants: {
      constantName: 1.0,
    },
  },
}
```

</details><br/>

Most of the time you wont need to specify ALL of that, you can rely on the defaults in many situations or a particular piece of state simply won't apply. But it's still a lot to cram into one object!

### Where do all those values come from?

First off, there's a lot of values in a Render Pipeline that are informed the **rendering technique** your app is using. Color target formats, multisampling, and depth/stencil settings are all most likely dictated by the structure of your renderer and won't be dependent on any given object's vertex structure or material. This means you can easily control how many pipeline variants result from those particular values, and it's likely to be tightly correlated with how many different types of render passes (color, shadow, post-process, etc) your application uses.

Next up are values that are dependent on the **vertex data layout** of your geometry. As we've already seen from the code snippets above, these are values like the number and order of buffers and attributes, their strides, formats, and offsets, and the primitive topology. Every mesh that you render that formats its vertex buffer differently than the others will need it's own variant of a pipeline, even if everything else is the same. For this reason it's best if you can normalize the structure of your vertex data as much as possible, though that may be difficult depending on where your assets come from. We'll talk about this more below. Additionally, if the mesh is animated or has other specialized effects it's likely to need a different variant of the pipeline both for the additional vertex data streams and the animation logic in the shader.

Finally, the remainder of the values are likely to come from your **material**. It's common that a renderer may support multiple different types of materials which require entirely different shaders to achive. For example, a Physically Based Rendering (PBR) surface vs. one that is unaffected by lighting ("fullbright" or "unlit"). But within groups of the same type of material only a few flags should affect the pipeline definition. A material being double sided will determine the cull mode, for example, and materials that are partially transparent will affect the blend modes of the color targets, as well as maybe alpha-to-coverage settings. Generally these should be pretty minimal, though, and the majority of your material information should be captured as texture or buffer data and supplied by a bind group.

It's worth mentioning that the code of the vertex and fragment shader modules sit in a strange place where they can be influenced by all three of those aspects, which can make it seem like another vector for increasing the number of pipelines in use, but you can get away with suprisingly few variants of your shader code itself by relying more on branching and looping, something that GPUs have gotten quite good at over the years. Again, we'll talk about this more below.

## Part 3: Identifying duplicate pipelines

Up to this point the render pipelines created by our code have only really taken into account the vertex data layout (buffer layout and primitive topology). So that's where we'll start when looking for duplicate pipelines.

The way I approach this is with a very simplistic caching mechanism. First, we collect all of the arguments that the pipeline needs in order to render correctly and put them in a "pipeline arguments" object.

```js
function getPipelineArgs(primitive, buffers) {
  return {
    topology: gpuPrimitiveTopologyForMode(primitive.mode),
    buffers,
  };
}
```

Then those arguments get passed into the method that gets a pipeline for the primitive in question. At that point, in order to determine if we need to create a new pipeline we generate a "key" value that captures every value in our arguments object and check it against a map of previously created pipelines. If the map already contains the key then we know it's a compatible pipeline and we should re-use it!

Now, the way that you generate the pipeline key is up to you, as long as it captures every value that would cause a new pipeline to be required. Given that our pipeline args object already contains that, a really quick and effective way to generate a key is... Just serialize it as a JSON string!

Yeah, that feels kinda ugly. But it works! And it ensures that as you add new pipeline arguments in the future for more advanced rendering you don't forget to update your key generation code.

```js
// Our pipeline cache.
const pipelineGpuData = new Map();

function getPipelineForPrimitive(args) {
  // Creates a key that uniquely identifies this combination of topology and buffer layout.
  const key = JSON.stringify(args);

  // Check the pipeline cache to see if a pipeline with that key already exists.
  let pipeline = pipelineGpuData.get(key);
  if (pipeline) {
    return pipeline;
  }

  // In no compatible pipeline exists, create a new one.
  const module = getShaderModule();
  pipeline = device.createRenderPipeline({
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: args.buffers,
    },
    primitive: {
      topology: args.topology,
    },
    // ...Other properties omitted to avoid clutter.
  });

  // Add the pipeline to the cache.
  pipelineGpuData.set(key, pipeline);

  return pipeline;
}
```

Nothing about the rest of the code should be too surprising. It's about as simple of a caching mechanism as you can get in JavaScript, and completely ignores more advanced needs like cache invalidation, but it'll do for the purposes of this document.

And now we've significantly reduced the number of pipelines that need to be created! In fact, if you make the above changes to the previous sample page you'll start seeing that most models you load only need a single pipeline now! That'll change when we start taking materials into account, and if we were handling things like animation it could add additional pipeline variations into the mix. But it highlights the fact that although glTF technically doesn't give you many guarantees about vertex layout, the reality is that within a single file the layout is usually going to be identical or extremely similar.

### Sorting attributes and buffers

So we found a bunch of geometry that can share the same pipeline, yay! We're done here, right?

Not quite, actually. If you ever want to display content from multiple files at once then you're in a situation where depending on the version of the tools that created the files, the flags they used, and the tools that initially created the asset in the first place you could end up with a variety of different (but valid) ways to represent effectively the same data. Consider the following two WebGPU buffer layouts for two different meshes loaded from two different files:

```js
const bufferLayout1 = [{
  arrayStride: 24,
  attributes: [{
    shaderLocation: 0, // Position
    format: 'float32x3',
    offset: 0
  }, {
    shaderLocation: 1, // Normal
    format: 'float32x3',
    offset: 12
  }]
}, {
  arrayStride: 8,
  attributes: [{
    shaderLocation: 2, // Texcoord
    format: 'float32x2',
    offset: 0
  }]
}]

const bufferLayout2 = [{
  arrayStride: 8,
  attributes: [{
    format: 'float32x2',
    shaderLocation: 2, // Texcoord
    offset: 0,
  }],
}, {
  arrayStride: 24,
  attributes: [{
    shaderLocation: 1, // Normal
    format: 'float32x3',
    offset: 12
  }, {
    shaderLocation: 0, // Position
    format: 'float32x3',
    offset: 0
  }],
}]
```

If you look carefully you can see that these in fact represent the same buffer layout! Our simple de-duping system from the previous code sample won't recognize that, though, because of a few differences:

 - Even though the same buffer layouts are used, the order is different.
 - Similarly, the postion and normal attributes are declared in a different order in the second layout.

So how can we improve that? Well, first off we should recognize that the order of the buffers in the pipeline descriptor _doesn't matter_ as long as at draw time we set the buffers in the corresponding slots. Similarly, the order that attributes are declared in doesn't have any effect on how the vertex data shows up in the shader, as long as the `shaderLocation` stays consistent. So we can increase the number of potential shared pipelines by sorting each buffer's attributes by `shaderLocation`, then sorting the buffers by the `attributes[0].shaderLocation`.

In practice it'll look something like this:

```js
function setupPrimitive(gltf, primitive) {
  // ...Omitting iteration through the primitive attributes, because it's unchanged.

  // During the attribute normalization step is a good place to sort the attributes.
  for (const buffer of bufferLayout.values()) {
    const gpuBuffer = gpuBuffers.get(buffer);
    for (const attribute of buffer.attributes) {
      attribute.offset -= gpuBuffer.offset;
    }
    // Sort the attributes by shader location.
    buffer.attributes = buffer.attributes.sort((a, b) => {
      return a.shaderLocation - b.shaderLocation;
    });
  }
  // Sort the buffers by their first attribute's shader location.
  const sortedBufferLayout = [...bufferLayout.values()].sort((a, b) => {
    return a.attributes[0].shaderLocation - b.attributes[0].shaderLocation;
  });

  // Ensure that the gpuBuffers are saved in the same order as the buffer layout.
  const sortedGpuBuffers = [];
  for (const buffer of sortedBufferLayout) {
    sortedGpuBuffers.push(gpuBuffers.get(buffer));
  }

 const gpuPrimitive = {
    buffers: sortedGpuBuffers,
    drawCount,
    instances: [],
  };
  const pipelineArgs = getPipelineArgs(primitive, sortedBufferLayout);
  primitiveGpuData.set(primitive, {
    // Make sure to pass the sorted buffer layout here.
    pipeline: getPipelineForPrimitive(pipelineArgs)
    buffers: sortedGpuBuffers,
    drawCount
  });
}
```

It should be noted that the previous work that we did to normalize attribute offsets *also* helps here! By normalizing the attribute offsets we create more opportunities for duplicate pipelines to be identified.

### Rethinking the render loop

Now we have most or all of our geometry using a single pipeline, but if we continue using the same render loop from above that actually doesn't help us much, because we're still setting the pipeline for every primitive we draw. If you're lucky the fact that you're simply setting same pipeline over and over again might be recognized by the driver and optimized, but WebGPU implementations have zero obligation to identify duplicate state changes for you. It will always be a far better strategy to optimize your own code to reduce unnecessary work than hope that some other part of the stack will magically make things faster for you.

You could do something like keep track of the last pipeline you set and compare it to the next one and then skip the `setPipeline()` call if they're the same, which would work for the samples linked on this page but would be unpredictable in more real-world situations.

Instead, we can get better, more predictable results by flipping our render loop on it's head. Currently, our render loop looks roughly like this:

 * For each node with a mesh
   * Set node transform bind group
   * For each primitive of that mesh
     * Set primitive pipeline and buffers
     * Draw primitive

But let's look at the relationship of the objects in that loop. Each Pipeline can be shared between many primitives, and each Primitive/Mesh can be referenced by multiple nodes. So if we want to minimize state changes, we probably want our render loop to look more like this:

 * For each pipeline
   * Set pipeline
   * For each primitive that uses pipeline
     * Set primitive buffers
     * For each node that references that primitive
       * Set node transform bind group
       * Draw primitive

That way while the number of times we set the node data stays the same the number of times we set the pipeline and primitive data has the potential to be drastically lower.

### Track your render data carefully.

In order to efficiently iterate through our render loop's data in the order prescribed above, we'll want to start tracking it differently. Remember, the more work we do up-front, the less we'll need to do at draw time. In this case that means saving our GPU data in a way that mimics our intended draw order.

First, we used to track a big list of node transforms and meshes. But that's going to be difficult to use if we're trying to look up which transforms to use for a primitive rather than visa-versa. So instead we should start tracking a list of transforms to be applied on every primitive. Let's call those "instances" of the primitive, shall we?

```js
// We don't need this map to persist into the draw loop, so we'll declare it here and pass
// it into the functions that need it.
const primitiveInstances = new Map();

for (const node of gltf.nodes) {
  if ('mesh' in node) {
    setupMeshNode(gltf, node, primitiveInstances);
  }
}

for (const mesh of gltf.meshes) {
  for (const primitive of mesh.primitives) {
    setupPrimitive(gltf, primitive, primitiveInstances);
  }
}
```

When setting up the mesh nodes now we want to build up a list of transform bind groups for each primitive.

```js
function setupMeshNode(gltf, node, primitiveInstances) {
  // ...Omitted bind group creation, since it's unchanged.

  // Loop through every primitive of the node's mesh and append this node's transform bind
  // group to the primitive's instance list.
  const mesh = gltf.meshes[node.mesh];
  for (const primitive of mesh.primitives) {
    let instances = primitiveInstances.get(primitive);
    if (!instances) {
      instances = [];
      primitiveInstances.set(primitive, instances);
    }
    instances.push(bindGroup);
  }
}
```

Also, previously we would store the pipeline to be used for each primitve in that same structure. But now we should start saving a list of primitives with every pipeline, like so:

```js
function getPipelineForPrimitive(args) {
  // ...Omitted pipeline deduplication and creation, since it's unchanged.

  const gpuPipeline = {
    pipeline,
    primitives: [] // Start tracking every primitive that uses this pipeline.
  };

  pipelineGpuData.set(key, gpuPipeline);

  return gpuPipeline;
}
```

And finally, when it's time to set up the primitives we want to both store the instance list that was created when iterating over the nodes as part of the GPU data that we need, and we want to push this primitive onto the list of primitives for the pipeline it uses.

```js
function setupPrimitive(gltf, primitive, primitiveInstances) {
  // ...Omitted buffer layout handling because it's unchanged

  const gpuPrimitive = {
    buffers: sortedGpuBuffers,
    drawCount,
    // Start tracking every transform that this primitive should be rendered with.
    instances: primitiveInstances.get(primitive),
  };

  const pipelineArgs = getPipelineArgs(primitive, sortedBufferLayout);
  const pipeline = getPipelineForPrimitive(pipelineArgs);

  // Don't need to link the primitive and gpuPrimitive any more, but we do need
  // to add the gpuPrimitive to the pipeline's list of primitives.
  pipeline.primitives.push(gpuPrimitive);
}
```

Now we've finished rearranging our data storage, so we can finally refactor our render loop:

```js
function renderGltf(renderPass) {
  renderPass.setBindGroup(0, frameBindGroup);

  for (const gpuPipeline of pipelineGpuData.values()) {
    renderPass.setPipeline(gpuPipeline.pipeline);

    for (const gpuPrimitive of gpuPipeline.primitives) {
      for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
        renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
      }

      for (const bindGroup of gpuPrimitive.instances) {
        renderPass.setBindGroup(1, bindGroup);
        renderPass.draw(gpuPrimitive.drawCount);
      }
    }
  }
}
```

You can see that this isn't much less complex than the previous version, we're just looping through it in a different order. But that simple change in the order of operations has a big impact on how much work we're ultimately doing, which hopefully you can see in the third sample page:

[![Sample 3 screenshot](/assets/images/sample-03.jpg)
Click to launch **Sample 03 - Pipeline Caching**](03-pipeline-caching.html)

For example, look at the "buggy" model on this page. In the prevous sample, we were calling `setPipeline()` 236 times every frame. With the changes described above we're now calling it once! And we're setting buffers less too, 444 times vs 708 in the previous sample.

A tradeoff is that we're calling `setBindGroup()` more now (237 times vs 191 in the previous sample), so we've traded off significantly less `setPipeline()` calls for more `setBindGroup()` calls. But that's OK! `setBindGroup()` is generally a cheaper operation AND we can use some additional tricks to reduce those calls too.

## Part 4: Instancing

Now, I'm sure that at least some readers were shouting at their screens at the end of the last section that the code wasn't doing *real* instancing. And they'd be right! Let's fix that.

If you're not familiar with the concept, Instancing in a graphics API is when you draw the same mesh multiple different times with small variants between them (like the transform) with a single draw call. This has been around since WebGL 1.0 in the form of an extension, but in WebGPU instancing is a core part of the API. So much so, in fact, that EVERY draw call in WebGPU is an instanced draw call! It's just that default number of instances when you call `draw()` or `drawIndexed()` is 1.

### Shader and buffer changes

When drawing instanced geometry, you need to provide something that communicates which data is different for each instance (otherwise why are you drawing the same thing over and over again?) There's two ways to do this: Either as a vertex buffer using `stepMode: 'instance'` in the pipeline's vertex state, or as a uniform or storage buffer that you index into using a per-instance index in the shader. For this document we'll take the latter approach since it maps a bit better to what we've already been doing.

The concept is pretty simple. Previously in the vertex shader we were using a uniform buffer to communicate the model matrix for every draw call, as shown in this simplified shader:

```rust
@group(1) @binding(0) var<uniform> modelMatrix : mat4x4<f32>;

@vertex
fn vertexMain(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
  // Omitting things like applying the view and projection transforms for simplicity.
  return modelMatrix * vec4(position, 1.0);
}
```

All we need to do to take advantage of instancing is change that `modelMatrix` uniform from a single matrix into an array of them, then use the WGSL builtin `instance_index` value to index into it.

```rust
@group(1) @binding(0) var<storage> modelMatrices : array<mat4x4<f32>>;

@vertex
fn vertexMain(@location(0) position : vec3<f32>,
              @builtin(instance_index) instance : u32) -> @builtin(position) vec4<f32> {
  return modelMatrices[instance] * vec4(position, 1.0);
}
```

Easy, right? And that ALMOST works with zero other code changes. There is one thing that we need to change in our WebGPU setup code.

You can see that between those two code snippets the binding type changed from `var<uniform>` to `var<storage>`, which means that the buffer we bind to it needs to change it's usage from `GPUBufferUsage.UNIFORM` to `GPUBufferUsage.STORAGE`.

Now, technically we CAN still use uniform buffers for our instance data. After all, uniform buffers can contain arrays, and the `instance_index` is just a regular `u32` value. The reason we're making the switch to a storage buffer here is that storage buffers allow for what's called "runtime sized arrays". That is, an array that doesn't have a specified length in the shader. Instead they implicitly let you index into as many elements as the storage buffer binding can contain.

In contrast, arrays into uniform buffers are required to have a fixed number of elements. This can work for instancing if, for instance, you know that you'll never render more than 100 instances of any given mesh at a time, but then you have to create each of your uniform buffers with enough space to define 100 instances worth of data and that's probably wasteful for most apps.

### Gathering transforms

After we make the above changes we still need to do some work to begin handling our transform data differently before we can start actually rendering more than once instance at a time.

To do this, we're going to change up the `setupMeshNodes` function to no longer create bind groups, and instead just start collecting the transforms associated with each primitive.

```js
function setupMeshNode(gltf, node, primitiveInstances) {
  // Loop through every primitive of the node's mesh and append this node's transform to
  // the primitives instance list.
  const mesh = gltf.meshes[node.mesh];
  for (const primitive of mesh.primitives) {
    let instances = primitiveInstances.get(primitive);
    if (!instances) {
      instances = [];
      primitiveInstances.set(primitive, instances);
    }
    instances.push(node.worldMatrix);
  }
}
```

We still _need_ the bind groups, of course! One way to approach it is to create them at the same time we create the rest of the GPU data for our primitive, after we've collected all the `primitiveInstances` from the nodes.

```js
function setupPrimitiveInstances(primitive, primitiveInstances) {
  // Get the list of instance transform matricies for this primitive.
  const instances = primitiveInstances.get(primitive);

  const count = instances.length;

  // Create a buffer large enough to contain all the instance matrices.
  const instanceBuffer = this.device.createBuffer({
    size: 16 * Float32Array.BYTES_PER_ELEMENT * count,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  // Loop through each instance and copy it into the instance buffer.
  const instanceArray = new Float32Array(instanceBuffer.getMappedRange());
  for (let i = 0; i < count; ++i) {
    instanceArray.set(instances[i], i * 16);
  }
  instanceBuffer.unmap();

  // Create a single bind group
  const bindGroup = device.createBindGroup({
    layout: instanceBindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: instanceBuffer },
    }],
  });

  // Return the bindGroup and the number of instance transforms it contains.
  return { bindGroup, count };
}

function setupPrimitive(gltf, primitive, primitiveInstances) {
  // ...Omitting everything else.

  const gpuPrimitive = {
    buffers: sortedGpuBuffers,
    drawCount,
    // Save the bind group and instance count.
    instances: setupPrimitiveInstances(primitive, primitiveInstances),
  };

  // ...
}
```

And now we've gathered all the transforms needed for each primitive into a single bind group for that primitive.

### Drawing instances

Finally, in the render loop, we can take our `draw()` call out of the innermost for loop it was in to iterate over the instances. Instead we only have to set the bind group once per primitive and then pass the number of instances to draw to the `draw()` call's [`instanceCount` argument](https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-drawindexed-indexcount-instancecount-firstindex-basevertex-firstinstance-firstindex)!

```js
function renderGltf(renderPass) {
  renderPass.setBindGroup(0, frameBindGroup);

  for (const gpuPipeline of pipelineGpuData.values()) {
    renderPass.setPipeline(gpuPipeline.pipeline);

    for (const gpuPrimitive of gpuPipeline.primitives) {
      for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
        renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
      }

      // Render every instance of this primitive in a single call.
      renderPass.setBindGroup(1, gpuPrimitive.instances.bindGroup);
      renderPass.draw(gpuPrimitive.drawCount, gpuPrimitive.instances.count);
    }
  }
}
```

For glTF models which take advantage of the ability for multiple nodes to reference a single mesh, this will now handle rendering of those case much more efficiently! In essence the loop that was here is still happening, but now we've pushed it down to the GPU's driver and given it a more efficient way to lookup the instance data. That can be a big performance win!

"But wait!" you might say. "That depends on the structure of the glTF file and/or the way the artist prepared the model. There's lots of files out there that won't take advantage of this. What about them?"

Well, good news! Even assuming that the file you load up has no opportunities for instancing at all, this approach is no worse than the previous version of the code. After all, previously we were creating a bunch of bind groups with only one transform in them and binding them one at a time, right? Well that's exactly what will happen in this code if no meshes are re-used. There's no penalty for passing `1` into the `instanceCount` of the `draw()` method. As mentioned earlier, that's the default anyway.

### But we can still do better!

Even in the case that we have no meshes that can be instanced, though, we can still use instancing to improve our render loop!

With the above changes, we've gone from creating a buffer for each transform to creating a buffer for all the transforms for a given primitive. But why stop there? Since the format of the transform data is the same for every mesh we're rendering, we might as well put _all_ of the transforms for _all_ of the primitives into one big buffer!

This requires a tiny bit more planning, but ultimately it's a minor change to the code we've already got. First off, we'll need to start storing more information in our `primitiveInstances` than just the matrices.

```js
const primitiveInstances = {
  matrices: new Map(), // The instance matrices for each primitive.
  total: 0,            // The total number of instance matrices.
  arrayBuffer: null,   // The array buffer that the matrices will be placed in.
  offset: 0,           // The offset (in matrices) of the last matrix written into arrayBuffer.
};
```

When collecting the matrices from the nodes, the only real change is that we need to start tracking the total number
of matrices that we encounter.

```js
function setupMeshNode(gltf, node, primitiveInstances) {
  const mesh = gltf.meshes[node.mesh];
  for (const primitive of mesh.primitives) {
    let instances = primitiveInstances.matrices.get(primitive);
    if (!instances) {
      instances = [];
      primitiveInstances.matrices.set(primitive, instances);
    }
    instances.push(node.worldMatrix);
  }
  // Make sure to add the number of matrices used for this mesh to the total.
  primitiveInstances.total += mesh.primitives.length;
}
```

And then after collecting all of the matrices in the scene create a single buffer big enough to contain all of them.

```js
// Create a buffer large enough to contain all the instance matrices for the entire scene.
const instanceBuffer = device.createBuffer({
  size: 16 * Float32Array.BYTES_PER_ELEMENT * primitiveInstances.total,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});

// Map the instance matrices buffer so we can write all the matrices into it.
primitiveInstances.arrayBuffer = new Float32Array(instanceBuffer.getMappedRange());
```

Now we setup all the primitives in the scene like usual, but when it's time to setup the instances for that primitive we write the matrices into the bigger buffer instead of creating a new one for each primitive. This requires us to track the offset of the last set of matrices that were written so that they don't overlap, and we'll need to store that offset as part of the primitives instance data (instead of the bind group).

It's worth noting that it's important to make sure that all the matrices for a single primitive are adjacent in the buffer or else instancing won't work properly.

```js
function setupPrimitiveInstances(primitive, primitiveInstances) {
  // Get the list of instance transform matricies for this primitive.
  const instances = primitiveInstances.matrices.get(primitive);

  const first = primitiveInstances.offset;
  const count = instances.length;

  // Place the matrices in the instances buffer at the given offset.
  for (let i = 0; i < count; ++i) {
    primitiveInstances.arrayBuffer.set(instances[i], (first + i) * 16);
  }

  // Update the offset for the next primitive.
  primitiveInstances.offset += count;

  // Return the index of the first instance and the count.
  return { first, count };
}
```

Once we've written all of the instance matrices for all of the primitives into the big instance buffer, we unmap it, which sends all the data we just placed in the array buffer to the GPU.

```js
// Unmap the buffer when we're finished writing all the instance matrices.
instanceBuffer.unmap();
```

So now all the matrices used by our scene are in one big buffer! But how do we tell each draw call what part of that buffer to use? You _could_ do it with bind groups. One option is creating a new bind group for each primitve and setting the `resource.offset` to the appropriate point in the buffer when defining the bind group's buffer entry. Another approach would be to create a single bind group with [dynamic offsets](https://gpuweb.github.io/gpuweb/#dom-gpubufferbindinglayout-hasdynamicoffset) for the buffer binding, which would then allow you to set the offset into the buffer when you call `setBindGroup()`.

Either of those approaches, though, will require you to call `setBindGroup()` once per primitive, which isn't really an improvement on what we had before. Instead, we can use an instancing trick that allows us to only set the bind group once. First, we create a single bind group that covers the entire instance buffer.

```js
// Create a bind group for the instance buffer.
const instanceBindGroup = device.createBindGroup({
  layout: instanceBindGroupLayout,
  entries: [{
    binding: 0, // Instance storage buffer
    resource: { buffer: instanceBuffer },
  }],
});
```

In the render loop, we bind that new bind group once at the very beginning of the loop. Then for every `draw()` call we pass one new piece of information: The offset into the buffer (in matrices) as the [`firstInstance` argument](https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-draw-vertexcount-instancecount-firstvertex-firstinstance-firstinstance).

This changes the value of the [`@builtin(instance_index)`](https://gpuweb.github.io/gpuweb/wgsl/#builtin-values) in the shader. For example, normally if we call draw with an `instanceCount` of 4 and leave the `firstInstance` as its default (0), the `instance_index` for each primitive would be `0, 1, 2, 3`. But if we set the `firstInstance` to 12, then the `instance_index` for each primitive will be `12, 13, 14, 15`.

Since we're using the `instance_index` to index into our array of matrices, you can see that we can use this value to indicate what offset into that array we should start at!

```js
function renderGltf(renderPass) {
  renderPass.setBindGroup(0, frameBindGroup);

  // Set the bind group containing all of the instance transforms.
  renderPass.setBindGroup(1, instanceBindGroup);

  for (const gpuPipeline of pipelineGpuData.values()) {
    renderPass.setPipeline(gpuPipeline.pipeline);

    for (const gpuPrimitive of gpuPipeline.primitives) {
      for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
        renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
      }

      // Every time we draw, pass an offset (in instances) into the instance buffer as the
      // "firstInstance" argument. This will change the initial instance_index passed to the
      // shader and ensure we pull the right transform matrices from the buffer.
      renderPass.draw(gpuPrimitive.drawCount, gpuPrimitive.instances.count, 0, gpuPrimitive.instances.first);
    }
  }
}
```

By doing this, we eliminate a `setBindGroup()` call for every primitive we draw in favor of simply passing in another argument to `draw()`, which is definitely a win from the perspective of JavaScript overhead for our rendering! (The 0 in between the `instanceCount` and `firstInstance` args is the `baseVertex`, which can be useful if you're packing all the geometry for your meshes into a larger buffers, but which we won't be using in these samples.)

(Fun fact: This trick can't be done in WebGL or WebGL 2, as it doesn't have an equivalent to the `firstInstance` argument. It's in development as a [WebGL 2 extension](https://www.khronos.org/registry/webgl/extensions/WEBGL_draw_instanced_base_vertex_base_instance/), though!)

[![Sample 4 screenshot](/assets/images/sample-04.jpg)
Click to launch **Sample 04 - Instancing**](04-instancing.html)

If we take a look at our sample app with these new modifications, we can now see that for some models, such as the "flight_helmet", the benefit of these changes is modest: We go from 7 bind group sets to 2, but the number of draw calls is the same because there's no repeated meshes.

For other models, like "sponza", the difference is significant! Wheras previously we were calling `draw()` 124 time, because this model has many repeated elements that share the same mesh, the new version of the code only calls `draw()` 33 times! And the number of bind group sets has gone from 125 to 2. Not bad at all!

### Knowing when to not pack everything into One Big Buffer

For our particular case, a simple static glTF renderer, the above strategy of packing all of our transforms into a single buffer works out great. But there are scenarios in which it wouldn't be as beneficial, or possibly even detrimental. Even though we won't be implementing any of those scenarios as part of this document, it's worth considering them for more real-world use cases.

The first thing to consider is if any of the transforms are going to be changing freqeuently. If any parts of the scene are animated you may want to consider placing their transforms in a separate buffer to make per-frame updates easier/faster. Similarly, if you have skinned meshes as part of your scene some of the instancing tricks we just covered may be more difficult to pull off, and having a separate code path for managing and rendering them may be warranted.

Another scenario to consider is if the contents of your scene are changing rapidly. If meshes are being added and removed all the time it's not practical to always allocate buffers with exactly the right amount of instance storage, as you'd end up re-allocating and re-populating the buffer almost every frame. A similar problem may emerge when using something like frustum culling, where the meshes in your scene are largely static but which ones you are choosing to render changes frequently. A potentially better approach in those scenarios could be to allocate an instance buffer large enough to handle a reasonable upp limit on the number of meshes you can render at once and update it as needed. Or spread the instance data across several smaller buffers that are cheaper to allocate and destroy as needed.

Also, if some instanced meshes in your scene have additional per-instance data that doesn't apply to all of them (for example, a per-instance color), then using the `firstInstance` to provide offsets into the instance data buffers becomes trickier. At that point it may be worth either splitting your instance buffers up based on the type of data required for each instance, or finding a different pattern for managing instances altogether.

Remember what I said at the beginning of this doc: There's rarely a "perfect" solution for any given problem, only a solution that works well with the tradeoffs your app is willing to make.

## Part 5.0: glTF Material overview

Up till this point we've been ignoring materials for the sake of simplicity, but now that we've got some of the quirks of rendering the geometry figured out, it's a good time to start looking at how to incorporate materials into our rendering.

Fair warning: This document has no interest in becoming a full-fledged tutorial on writing a good PBR shader or anything like that. All we'll really be doing is incorporating the materials base color and any properties that affect how the pipeline is created. I happily leave the process of extending those patterns to the rest of glTF's material properties as an exercise for the reader.

Materials in glTF are generally comprised of four different glTF properties: `images`, `samplers`, `textures`, and `materials`

### Images

There's a bit of terminology confusion here, because glTF `images` map most closely to WebGPU Textures, but otherwise it's a pretty straightforward thing to load them. Unless you're using extendions, glTF defines that all images come in the form of a JPEG or PNG file, either defined by a relative URI or encoded in one of the binary buffers. That's great news for us, because browsers happen to be really good at loading image files! We just have to push the contents of the images that the browser loads for us into a texture.

I've written an entirely separate document about best practices for [creating WebGPU textures from image elements](../img-textures.md). It even has a section _specifically about loading images from a glTF file!_ So really, just go read that. It won't take long. Because the code is short, however, I'll stick it here for reference.

```js
async function createTextureForGltfImage(gltf, image) {
  let blob;
  if (image.uri) {
    // Image is given as a URI
    const response = await fetch(image.uri);
    blob = await response.blob();
  } else {
    // Image is given as a bufferView.
    const bufferView = gltf.bufferViews[image.bufferView];
    const buffer = getArrayBufferForGltfBuffer(bufferView.buffer);
    blob = new Blob(
      [new Uint8Array(buffer, bufferView.byteOffset, bufferView.byteLength)],
      { type: image.mimeType }
    );
  }
  const imgBitmap = await createImageBitmap(blob);

  const descriptor = {
    size: { width: imgBitmap.width, height: imgBitmap.height },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  };
  const texture = device.createTexture(descriptor);
  device.queue.copyExternalImageToTexture({ source: imgBitmap }, { texture }, descriptor.size);

  // Mipmap generation omitted for simplicity. Seriously, though, go read the doc I linked above!

  return texture;
}
```

### Samplers

WebGPU Samplers define how data is read out of a texture. Is it filtered? Does it repeat? Etc. glTF's `samplers` are defined in WebGL terms, but fortunately they translate very cleanly into WebGPU with minimal effort.

The only quirk that should be pointed out is that [the spec says](https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#_overview) that any time the `minFilter` and `magFilter` properties are undefined the implementation is free to apply whatever filtering it wishes. 99.999% of the time you'll want to just default to `'linear'` filtering in those cases, and set the `mipmapFilter` to `'linear'` as well unless you can't generate mipmaps for some reason.

```js
function gpuAddressModeForWrap(wrap) {
  switch (wrap) {
    case WebGLRenderingContext.CLAMP_TO_EDGE: return 'clamp-to-edge';
    case WebGLRenderingContext.MIRRORED_REPEAT: return 'mirror-repeat';
    default: return 'repeat';
  }
}

function createSamplerForGltfSampler(sampler) {
  const descriptor = {
    addressModeU: gpuAddressModeForWrap(sampler.wrapS),
    addressModeV: gpuAddressModeForWrap(sampler.wrapT),
  };

  // WebGPU's default min/mag/mipmap filtering is nearest, se we only have to override it if we
  // want linear filtering for some aspect.
  if (!sampler.magFilter || sampler.magFilter == WebGLRenderingContext.LINEAR) {
    descriptor.magFilter = 'linear';
  }

  switch (sampler.minFilter) {
    case WebGLRenderingContext.NEAREST:
      break;
    case WebGLRenderingContext.LINEAR:
    case WebGLRenderingContext.LINEAR_MIPMAP_NEAREST:
      descriptor.minFilter = 'linear';
      break;
    case WebGLRenderingContext.NEAREST_MIPMAP_LINEAR:
      descriptor.mipmapFilter = 'linear';
      break;
    case WebGLRenderingContext.LINEAR_MIPMAP_LINEAR:
    default:
      descriptor.minFilter = 'linear';
      descriptor.mipmapFilter = 'linear';
      break;
  }

  return device.createSampler(descriptor);
}
```

The only other thing to be aware of is that in some cases (actually, a lot of cases!) no sampler will be specified. In those situations you'll need to use a default sampler, which is even easier:

```js
function createDefaultGltfSampler() {
  return device.createSampler({
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    minFilter: 'linear',
    magFilter: 'linear',
    mipmapFilter: 'linear',
  });
}
```

### Textures

In glTF `textures` are a combination of an image and a sampler. If this seems strange to you, blame WebGL. (In WebGL 1.0/OpenGL ES 2.0 you set the sampler properties directly on the texture.) We don't need to do anything special to handle them in our renderer, it's just an extra level of indirection to bounce through when looking up data.

### Materials

And now we finally can look at the glTF `materials` themselves. Without any extensions, there's only one type of material that glTF supports and that's the `pbrMetallicRoughness` model. This refers to a Physically Based Rendering (PBR) model where the surface is described in terms of how much it resembles a metal and how rough the surface is on a scale from 0 to 1. It can also define a base color and emissiveness for the surface, a normal and an occlusion map. Finally, there's some properties that control how the geometry is blended and culled.

Here's a relatively complete example:

```json
"materials": [{
 "pbrMetallicRoughness": {
    "baseColorTexture": { "index": 1 },
    "baseColorFactor": [ 1.0, 0.75, 0.35, 1.0 ],
    "metallicRoughnessTexture": { "index": 5 },
    "metallicFactor": 1.0,
    "roughnessFactor": 0.0
  },
  "normalTexture": { "index": 2 },
  "occlusionTexture": {
    "index": 4,
    "strength": 0.9
  },
  "emissiveTexture": { "index": 3 },
  "emissiveFactor": [0.4, 0.8, 0.6],
  "alphaMode": "OPAQUE",
  "doubleSided": true,
}],
```

## Part 5.1: Applying material properties

Now that we've got a good idea of what makes up a glTF material we need to start applying those properties to our rendering. This needs to happens in accross several parts of the code.

### Alpha blending and culling

The first part of the materials that we can apply to our rendering is the `doubleSided` and `alphaMode` properties. These both affect values in the render pipline descriptor, and as a result different values here will result in new pipeline variants, even if the geometry layout is identical.

Fortunately our previous code changes have prepared us very well for this! First, we'll want to start passing the material when we generate the pipeline args so that it can contribute to determining if a new pipeline is needed.

```js
function getPipelineArgs(topology, buffers, material) {
  return {
    topology: gpuPrimitiveTopologyForMode(primitive.mode),
    buffers,
    doubleSided: material.doubleSided,
    alphaMode: material.alphaMode,
  });
}
```

Next, when creating the pipeline, we need to set the culling and blending properties accordingly.

Handling the culling is easy, it's literally one line in the `createRenderPipeline()` call:

```js
// In getPipelineForPrimitive():

pipeline = device.createRenderPipeline({
  primitive: {
    topology,
    // Make sure to apply the appropriate culling mode
    cullMode: args.doubleSided ? 'none' : 'back',
  },
  // ... Other values omitted for brevity.
});
```

The `alphaMode` is a bit trickier, but not by much. If the requested mode is `"BLEND"` then we need to set the color blend factors of the pipeline accordingly.

```js
// In getPipelineForPrimitive():

// Define the alpha blending behavior.
let blend = undefined;
if (args.alphaMode == "BLEND") {
  blend = {
    color: {
      srcFactor: 'src-alpha',
      dstFactor: 'one-minus-src-alpha',
    },
    alpha: {
      // This just prevents the canvas from having alpha "holes" in it.
      srcFactor: 'one',
      dstFactor: 'one',
    }
  }
}

pipeline = device.createRenderPipeline({
  fragment: {
    targets: [{
      format: cavasColorFormat,
      // Apply the necessary blending
      blend,
    }],
  },
  // ... Other values omitted for brevity.
});
```

You may notice that this doesn't account for glTF's `"MASK"` blend mode, but we'll get to that in a bit.

Otherwise, that's the entirity of the changes we need to make to the pipelines to support materials! The rest of the material values will be communicated either via bind groups or shader changes.

<details markdown=block>
  <summary markdown=span><b>A note on transparent surface ordering</b></summary>
  One you start rendering models with alpha blending, you're bound to start seeing some artifacts due to the order the geometry is rendered in. In short: If transparent geometry is rendered before something that should appear behind it, depth testing may cull away that geometry leaving a hole when you look at it through the transparent surface.

  Traditionally the way to handle this is to render all opaque surfaces first, then render all transparent surfaces sorted back-to-front from the camera's point of view. This still isn't perfect, though, as you could have very large transparent meshes that don't sort trivailly.

  There's also newer methods for [order-independent transparancy](https://en.wikipedia.org/wiki/Order-independent_transparency) that typically involve storing data about transparent surfaces in intermediate render targets and resolving them in the correct order as a post process. These can achive good results but are more expense in terms of both rendering and memory usage and typically not favored for performance sensitive applications like games, especially on mobile GPUs.

  This document's approach to correct transparency rendering is to stick its fingers in its ears and loudly shout _"LA LA LA! CAN'T HEAR YOU!"_ while carefully picking models that don't exhibit the problem.
</details>

### Managing material bind groups

The remainder of the material values are generally all either scalar and vector values that need to be exposed as uniforms, or textures. In order to communicate these to the shader, we'll need to create a Bind Group for each material.

In order to track the material bind groups, we'll need to employ what, by now, is probably a familiar pattern. Maintain a map of glTF `materials` to the appropriate WebGPU data, populate it for each `material` in the model, and then associate the right one with each `primitive` that we process.

```js
const materialGpuData = new Map();

for (const material of gltf.materials) {
  setupMaterial(gltf, material, materialGpuData);
}

// Omitting Instance setup...

for (const mesh of gltf.meshes) {
  for (const primitive of mesh.primitives) {
    setupPrimitive(gltf, primitive, primitiveInstances, materialGpuData);
  }
}
```

For each material we need to populate a uniform buffer with the material's scalar and vector values, and then create a bind group pointing to both that buffer and the appropriate textures.

```js
function setupMaterial(gltf, material, materialGpuData) {
  // Create a uniform buffer for this material and populate it with the material properties.
  // For these samples we're only doing partial material support, so we only need room for
  // 5 floats. (4 for baseColorFactor and 1 for alphaCutoff)
  const materialUniformBuffer = device.createBuffer({
    size: 5 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const materialBufferArray = new Float32Array(materialUniformBuffer.getMappedRange());

  // Place the material values into the buffer, using defaults if they're not specified.
  materialBufferArray.set(material.pbrMetallicRoughness?.baseColorFactor || [1, 1, 1, 1]);
  materialBufferArray[4] = material.alphaCutoff || 0.5;
  materialUniformBuffer.unmap();

  // The baseColorTexture may not be specified either. If not use a plain white texture instead.
  let baseColor = gltf.gpuTextures[material.pbrMetallicRoughness?.baseColorTexture?.index];
  if (!baseColor) {
    baseColor = {
      texture: opaqueWhiteTexture,
      sampler: defaultSampler,
    };
  }

  // Create a bind group with the uniform buffer, base color texture, and sampler.
  const bindGroup = this.device.createBindGroup({
    label: `glTF Material BindGroup`,
    layout: this.materialBindGroupLayout,
    entries: [{
      binding: 0, // Material uniforms
      resource: { buffer: materialUniformBuffer },
    }, {
      binding: 1, // Sampler
      resource: baseColor.sampler,
    }, {
      binding: 2, // BaseColor
      resource: baseColor.texture.createView(),
    }],
  });

  // Associate the bind group with this material.
  materialGpuData.set(material, {
    bindGroup,
  });
}
```

The `opaqueWhiteTexture` used as the default for the base color texture here should be as simple as possible: a single white pixel. It should created outside this method and shared between all materials that need it. The method for creating such a texture is pretty simple:

```js
function createSolidColorTexture(r, g, b, a) {
  const data = new Uint8Array([r * 255, g * 255, b * 255, a * 255]);
  const texture = device.createTexture({
    size: { width: 1, height: 1 },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });
  device.queue.writeTexture({ texture }, data, {}, { width: 1, height: 1 });
  return texture;
}
```

In additiona to that you'll likely find that you'll want to keep a transparent black texture for thing like emissive texture defaults and a default normal texture as well:

```js
const opaqueWhiteTexture = createSolidColorTexture(1, 1, 1, 1);
const transparentBlackTexture = createSolidColorTexture(0, 0, 0, 0);
const defaultNormalTexture = createSolidColorTexture(0.5, 0.5, 1, 1);
```

### Part 5.2: Associating materials with primitives

After that's been done make sure to associate the primitive with the bind group we just created during the primitive setup. We can start by simply adding the material properties to the primitve GPU data we're already tracking.

```js
setupPrimitive(gltf, primitive, primitiveInstances, materialGpuData) {
  // ...Omitted buffer setup, since it's unchanged.

  //
  const material = gltf.materials[primitive.material];

  const gpuPrimitive = {
    buffers: sortedGpuBuffers,
    drawCount,
    instances: setupPrimitiveInstances(primitive, primitiveInstances),
    // Save the bind group for the primitives material.
    material: materialGpuData.get(material),
  };

  // Start passing the material when generating pipeline args.
  const pipelineArgs = getPipelineArgs(primitive, sortedBufferLayout, material);
  const pipeline = getPipelineForPrimitive(pipelineArgs);
  pipeline.primitives.push(gpuPrimitive);
}
```

Which lets us start setting the material's bind group in the render loop.

```js
function renderGltf(renderPass) {
  renderPass.setBindGroup(0, frameBindGroup);
  renderPass.setBindGroup(1, instanceBindGroup);

  for (const gpuPipeline of pipelineGpuData.values()) {
    renderPass.setPipeline(gpuPipeline.pipeline);

    for (const gpuPrimitive of gpuPipeline.primitives) {
      for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
        renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
      }

      // Set the material bind group for this primitive.
      renderPass.setBindGroup(2, gpuPrimitive.material.bindGroup);

      renderPass.draw(gpuPrimitive.drawCount, gpuPrimitive.instances.count, 0, gpuPrimitive.instances.first);
    }
  }
}
```

And with that we're now passing all the information we need to begin rendering with materials! But if you were to run the code at this point you'd find that all of those changes have yet to make any difference to the rendered output because we've got one significant change left to make: Updating the shader.

## Part 5.2: Using materials in the shader

Writing shaders that efficiently render PBR materials like those used by glTF is, to put it mildly, a large topic. You can find volumes of writing from talented developers who have spent far more time researching the topic than I have with you favorite search engine, so I'm not even going to try to capture the full scope of it here.

Instead what I'm most interested in is covering some basic patterns that can be used when writing shaders so that they can handle a variety of common scenarios, which will hopefully allow you to easily expand the shaders to encompass material properties beyond the ones we cover here.

### Incorporating Uniform values

A simple place to start is applying a uniform value like the base color to our mesh. This uses the same principles that have already been in use for pulling in camera uniforms throughout these samples: You declare a struct that defines the layout of the values in the uniform butter, associate it with a particular group and binding index, and then reference the values in the shader.

A minimal example of using the base color factor may look like this:

<!-- Turns out Rust is a reasonable proxy for WGSL syntax highlighting -->
```rust
// Omitting vertex shader...

// Material struct mirrors how the data is packed in setupMaterial()
struct Material {
  baseColorFactor : vec4<f32>,
  alphaCutoff: f32,
};
// Group is the index specified set when calling setBindGroup(), binding is the
// index specified in the bindGroup/bindGroupLayout.
@group(2) @binding(0) var<uniform> material : Material;

struct VertexOutput {
  @location(0) normal : vec3<f32>
};

// Some hardcoded lighting
const lightDir = vec3(0.25, 0.5, 1.0);
const lightColor = vec3(1.0, 1.0, 1.0);
const ambientColor = vec3(0.1, 0.1, 0.1);

@fragment
fn fragmentMain(input : VertexOutput) -> @location(0) vec4<f32> {
  // Get the material's color value (stored as a variable to improve readability).
  let baseColor = material.baseColorFactor;

  // An extremely simple directional lighting model, just to give our model some shape.
  let N = normalize(input.normal);
  let L = normalize(lightDir);
  let NDotL = max(dot(N, L), 0.0);
  // Start factoring the base color into the lighting model.
  let surfaceColor = (baseColor.rgb * ambientColor) + (baseColor.rgb * NDotL);

  // Use the base color alpha as well.
  return vec4(surfaceColor, baseColor.a);
}
```

After making that change if we view models that use the `baseColorFactor`, such as the "buggy" model, you'll start seeing those colors come through in the rendering.

[![Buggy, now in color.](/assets/images/buggy-color.jpg)
Click to launch **Sample 05 - Materials**](05-materials.html?model=buggy)

Worth noting that a lot of models _don't_ use `baseColorFactor`, relying instead exclusively on textures for their coloring. But since our code in `setupMaterial()` provided a default base color of white when the base color is missing every model without a `baseColorFactor` continues to render exactly like it did before.

### Adding textures

Next we'll want to start making use of textures, which is also relatively simple, but requires us to start making use of a vertex attribute that thus far we've been ignoring: texture coordinates.

You may recall that way back at the beginning of this document we defined a JS object that mapped glTF attribute names to shader locations, and set up our buffer handling code to ignore anything that wasn't in that map. That means that the first step to exposing texture coordinates to our shader is to add the `TEXCOORD_0` attribute to that map, at which point the rest will be handled automatically with the buffer handling code we've already built!

```js
const ShaderLocations = {
  POSITION: 0,
  NORMAL: 1,
  // Add texture coordinates to the list of attributes we care about.
  TEXCOORD_0: 2
};
```

<details markdown=block>
  <summary markdown=span><b>A note on glTF texture coordinates</b></summary>
  glTF supports using multiple different sets of texture coordinates for a single primitive, denoted by the number at the end of the attribute name. `TEXCOORD_1`, `TEXCOORD_4`, etc. These will then correspond with the optional `texCoord` value specified in the [material's texture references](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-textureinfo).

  This can be useful for certain types of effects, but in my experience it's fairly rare that models actually make use of multiple texture coordinates in a meaningful way. As such, this document will be ignoring them for simlicity, but be aware that it's something that needs to be considerd if you are trying to make a strictly compliant glTF renderer.
</details>

Then the texture coordinate attribute can be accessed in the vertex shader and passed along to the fragment shader.

```rust
// Camera/model uniforms omitted for brevity...

struct VertexInput {
  @builtin(instance_index) instance : u32,
  @location(${ShaderLocations.POSITION}) position : vec3<f32>,
  @location(${ShaderLocations.NORMAL}) normal : vec3<f32>,

  // Newly added texcoord attribute.
  @location(${ShaderLocations.TEXCOORD_0}) texcoord : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,

  // Texcoord needs to be passed to the fragment shader as well.
  @location(1) texcoord : vec2<f32>,
};

@vertex
fn vertexMain(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;

  let modelMatrix = model[input.instance];
  output.position = camera.projection * camera.view * modelMatrix * vec4(input.position, 1.0);
  output.normal = normalize((camera.view * modelMatrix * vec4(input.normal, 0.0)).xyz);

  // Copy the texcoord to the fragment shader without change.
  output.texcoord = input.texcoord;

  return output;
}
```

Which in turn allows the fragment shader to sample from the `baseColorTexture` correctly.

```rust
@group(2) @binding(1) var materialSampler : sampler;
@group(2) @binding(2) var baseColorTexture : texture_2d<f32>;

@fragment
fn fragmentMain(input : VertexOutput) -> @location(0) vec4<f32> {
  // Get the combined base color from the texture and baseColorFactor.
  let baseColor = textureSample(baseColorTexture, materialSampler, input.texcoord) * material.baseColorFactor;

  // Omitting remaineder of fragment shader, which is unchanged...
}
```

And now we have textured models as well!

[![Sample 5 screenshot](/assets/images/sample-05.jpg)
Click to launch **Sample 05 - Materials**](05-materials.html)

So we're basically done here, right?

### Handling missing attributes

Not quite, as it turns out, because while the above change works great for models that include texture coordinates (which, to be fair, is a very common thing) we've no introduced errors for models that have none, such as the "buggy" model. If you try to load it now you'll see something like the following:

```
Pipeline vertex stage uses vertex buffers not in the vertex state
 - While validating vertex state.
 - While calling [Device].CreateRenderPipeline([RenderPipelineDescriptor]).
```

What this means is, effectively, that any attributes you specify in the vertex shader MUST have a corresponding attribute in the render pipeline's vertex state. As a result if you have a model that simply doesn't include one of the attributes your shader makes use of you have to compensate for it somehow to make the pipeline and shader states match.

There's a few ways that you can do that. One approach, for example, would be to generate a new buffer on the fly for any missing attributes. How well this works for you is likely a matter of how your app is structured and what data you need to generate.

For example, a missing texture coordinate can be accounted for by simply allocating a vertex buffer that's big enough. The default buffer contents of all zeros works well for an attribute that will be used to sample from a default 1 pixel texture. (If your model includes an actual texture but no texture coordinates it should be considered a broken file.)

There's a variant of the same trick where you can generate a much smaller buffer by specifying in the render pipeline's vertex state that the buffer's step mode is `"instance"`, at which point you only have to create a buffer large enough to contain one attribute value per instance. This saves quite a bit of memory, but works best when you know in advance how many instances of a given object could ever be rendered (or, at the very least, the maximum number of instances that your app will render at a time.)

If the attribute that you are missing is something like a normal, however, then blindly using a buffer full of zeros in it's place will result in broken lighting. You'd need to generate reasonable normals based off the model geometry, which is much more involved.

A different approach, and the one that we'll cover here, is to create a new variant of the shader that accounts for the missing atributes somehow. As a general rule you want to avoid shader variants as much as possible, for all the same reasons you want to avoid pipeline variants (and because every shader variant requires a pipeline variant), but sometimes it's simply the most effective approach.

### Caching shader variants

Our system for handling shader variants will piggyback on how we already handle pipeline variants. While generating the pipeline args we'll also generate a new sub-section of that object specifically for shader args, and use that as the map key for caching in our existing `getShaderModule()` function. It's important that the shader args ALSO be part of the pipeline args, because we want any changes in the shader to also be recognized as something that requires a pipeline variant.

```js
const shaderModules = new Map();

function getPipelineArgs(primitive, buffers, material) {
  return {
    topology: gpuPrimitiveTopologyForMode(primitive.mode),
    buffers,
    doubleSided: material.doubleSided,
    alphaMode: material.alphaMode,
    // These values specifically will be passed to shader module creation.
    shaderArgs: {
      hasTexcoord: 'TEXCOORD_0' in primitive.attributes,
      useAlphaCutoff: material.alphaMode == 'MASK',
    },
  };
}

function getShaderModule(args) {
  const key = JSON.stringify(args);

  let shaderModule = shaderModules.get(key);
  if (!shaderModule) {
    const code = `
      // Generate the shader module code...
    `;

    shaderModule = device.createShaderModule({ code });
    shaderModules.set(key, shaderModule);
  }

  return shaderModule;
}

function getPipelineForPrimitive(args) {
  // Majority of function omitted for brevity...

  const module = getShaderModule(args.shaderArgs);
}
```

### Composing shader variants

Now comes a slightly tricky part. How do we actually put those arguments to use when building the shader? In this case, we need to either add or omit the `@location() texcoord : vec2<f32>` from the vertex inputs depending on the value of `args.hasTexcoord`.

If you are coming to WebGPU from WebGL, you might be familiar with using GLSL preprocessor statements to conditionally switch code on an off and go looking for a similar mechanism in WGSL. Unfortunately you'll be unpleasantly suprised to learn that WGSL has no built-in preprocessor support.

Instead you'll have to rely on JavaScript's string manipulation mechanisms to get by. At it's simplest this could means basic string concatenation.

```js
let code = `
  struct VertexInput {
    @builtin(instance_index) instance : u32,
    @location(${ShaderLocations.POSITION}) position : vec3<f32>,
    @location(${ShaderLocations.NORMAL}) normal : vec3<f32>,
`;

if (args.hasTexcoord) {
  code += `@location(${ShaderLocations.TEXCOORD_0}) texcoord : vec2<f32>,`;
}

code += `
  };
`;
```

But shaders can be big, complex things! And if you're anything like me the idea of building out a large shader with code like that example feels pretty painful.

That's why I [built a simple library](https://github.com/toji/wgsl-preprocessor) to add preprocessor-like syntax to WGSL using JavaScript's [tagged templates](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals#tagged_templates). By tagging a template literal with `wgsl` it handles parsing `#if`, `#elif`, `#else`, and `#endif` statements, allowing the above code to look like this instead:

```js
import { wgsl } from 'https://cdn.jsdelivr.net/npm/wgsl-preprocessor@1.0/wgsl-preprocessor.js';

// Note that the string is prefixed with the wgsl tag
let code = wgsl`
  struct VertexInput {
    @builtin(instance_index) instance : u32,
    @location(${ShaderLocations.POSITION}) position : vec3<f32>,
    @location(${ShaderLocations.NORMAL}) normal : vec3<f32>,
    #if ${args.hasTexcoord}
      @location(${ShaderLocations.TEXCOORD_0}) texcoord : vec2<f32>,
    #endif
  };
`;
```

Which I find far easier to read and write! As a result I'll be using that syntax for the rest of the shader code, but feel free to use a different approach if it suits your needs!

In addition to the above change to the input attributes, we also need to handle any other location in the shader that might make use of the texture coordinates. I find that rather than omit every texture lookup from the shader it's easier to simply pass a default value to the fragment shader and allow the rest of the shader to continue operating as before.

```js
let code = wgsl`
  // Omitting most of the shader for brevity...

  struct VertexInput {
    @builtin(instance_index) instance : u32,
    @location(${ShaderLocations.POSITION}) position : vec3<f32>,
    @location(${ShaderLocations.NORMAL}) normal : vec3<f32>,
    #if ${args.hasTexcoord}
      @location(${ShaderLocations.TEXCOORD_0}) texcoord : vec2<f32>,
    #endif
  };

  @vertex
  fn vertexMain(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;

    let modelMatrix = model[input.instance];
    output.position = camera.projection * camera.view * modelMatrix * vec4(input.position, 1.0);
    output.normal = normalize((camera.view * modelMatrix * vec4(input.normal, 0.0)).xyz);

    #if ${args.hasTexcoord}
      output.texcoord = input.texcoord;
    #else
      // Use a default texcoord if an appropriate attribute is not provided.
      output.texcoord = vec2(0.0);
    #endif

    return output;
  }
`;
```

And with that change we can now render both textured _and_ untextured models!

### Masked alpha

I mentioned before that we'd handled glTF's `"MASK"` `alphaMode` later, and now we're in a great position to do it! When using the `"MASK"` mode we want to simply check to see if the base color's alpha is below the threshold specified by the material's `alphaCutoff` property. If it is, we call `discard` in the fragment shader to stop the fragment from being rendered.

We've already passing the `alphaCutoff` into the shader as a uniform, but we only want to actually check against it if the `alphaMode` was `"MASK"`, which makes this another decent candidate for our shader preprocessor.

```js
let code = wgsl`
  @fragment
  fn fragmentMain(input : VertexOutput) -> @location(0) vec4<f32> {
    let baseColor = textureSample(baseColorTexture, materialSampler, input.texcoord) * material.baseColorFactor;

    #if ${args.useAlphaCutoff}
      // If the alpha mode is MASK discard any fragments below the alpha cutoff.
      if (baseColor.a < material.alphaCutoff) {
        discard;
      }
    #endif

    // Omitting most of the shader for brevity...
  }
`;
```

This change allows the plants in the "sponza" scene to render correctly, since they use `"MASK"` mode to hide the excess geometry around the leaves and flowers.

Before implementing `alphaMode: "MASK"`
[![Sponza's plants without alpha discard](/assets/images/no-alpha-mask.jpg)](05-materials.html?model=sponza)

After implementing `alphaMode: "MASK"`
[![Sponza's plants with alpha discard](/assets/images/alpha-mask.jpg)
Click to launch **Sample 05 - Materials**](05-materials.html?model=sponza)

### Don't go overboard with shader variants!

Now that we've got a system in place for creating shader variants whenever we need it, it can be very tempting to want to apply that to everything! For example, what about when a material doesn't include a particular texture channel? Why not create a new shader variant that omits the texture lookup! That feels like an obvious optimization opportunity (especially for things like normal, occlusion, or emissive maps). Unfortunately it's generally not a great idea.

Why? Well, first off, every shader variant that we create obviously requires a new render pipeline, and the more combinations of shader variants that can be created the more likely you are to have the number of shaders (and thus pipelines) you are using explode. We just spent a majority of this document finding ways to reduce the number of pipelines we're using! Let's not sabatoge that with overly aggressive shader customization now!

It's fairly unlikely that the cycles saved by, for instance, not sampling the occlusion map are going to make up for the overhead of more pipeline switching. It's often better to simply use a default texture and sample from it like usual, especially because the default textures _should_ be a single pixel, which means the value will be very effectively cached!

That said, there are still some scenarios where the overhead of fully computing some aspect of a material if you know that the material properties don't require it CAN be beneficial. [Unreal Engine](https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile), for example, has a "Fully Rough" variant of their PBR shaders that is used when an object has a constant roughness of 1 that cuts out a lot of reasonably expensive computation, especially for mobile devices.

Even when using a shader variant that doesn't use a given material value, however, you almost never want to omit that value from the bind group. There's no requirement that a given pipeline makes use of every value present in a bind group, so having a value present that's only used by some pipelines is OK. Bind group variants are yet another thing that forces new pipeline variants as well as introducing questions about whether or not a given bind group can be used in a given situation. As long as the material property can be given a cheap default value (such as a 1 pixel texture or a zeroed out uniform in a buffer) it's best to keep your bind groups as uniform as possible, as you'll find it leads not only to more efficient binding in the render loop but also more flexibility as you expand your renderer.

Times where you may want to consider creating bind group variants are when a given shader requires large bind group values that are not easily faked, such as skinning data.

## Part 5.3: One last render loop optimization

At this point we've taken the material rendering as far as this document is going to take it, but there's still one more thing we can look at before wrapping up.

Prior to the addition of materials our render loop was pretty tight, with a reasonably minimal amount of state setting to draw our scene. Now, however, we've added back in a per-primitive `setBindGroup()` for the material properties after working so hard to remove one in the instancing section earlier. Is there anything we can do about that?

Turns out, yes! One thing we should consider is that it's fairly common for models, especially larger ones, to share materials across multiple meshes/primitives. Think of, for instance, the "sponza" scene. That model contains 25 materials and 33 primitives (many of which are instanced). Obviously some of the primitves must use the same material.

Given this, it's likely that at least a few of our `setBindGroup()` calls are redundant. Would be nice to eliminate those, right?

Thinking through the chain of dependencies, any given material may have properties which change the render pipeline (`doubleSided`, `alphaMode`), but there's also a good chance that many materials will be able to share a pipeline (assuming the primitives using the material have compatible buffer layouts). There's a good chance that when all is said and done the number of materials in our scene will be greater than or equal to the number of pipielines, and the number of primitives will be greater than or equal to the number of materials. As such, ideally we'd like our render loop to look a bit more like this:

* For each pipeline
   * Set pipeline
   * For each material that uses pipeline
     * Set material bind group
     * For each primitive that uses material with pipeline
       * Draw primitive

And we can get exactly that with (you guessed it) MORE CACHING!

### Saving materials per-pipeline

The first change we need to make to get to that new render loop is to change which data we're saving with the `pipelineGpuData`. Currently we save both the `GPURenderPipeline` and an array of the `primitives` that use it. Now we should it so that each new `gpuPipeline` saves a map of gpuMaterials to an array of primitives that uses that material, like so:

```js
function getPipelineForPrimitive(args) {
  // Majority of function omitted for brevity...

  const gpuPipeline = {
    pipeline,
    // Cache a map of materials to the primitives that used them for each pipeline.
    materialPrimitives: new Map(),
  };

  pipelineGpuData.set(key, gpuPipeline);

  return gpuPipeline;
}
```

We'll populate that map in `setupPrimitive()`. Every time we get the primitives pipieline we'll look up the primitive's current `gpuMaterial` in the map. If it's not present we'll add a new array to the map with the `gpuMaterial` as the key. The primitives using that material will then be pushed onto the array. It's very close to the previous system, with just one extra level of indirection to allow us to save the material relationship. (This change also means we can stop saving the material on the `gpuPrimitive` itself.)

```js
function setupPrimitive(gltf, primitive, primitiveInstances, materialGpuData) {
  // Majority of function omitted for brevity...

  const gpuPrimitive = {
      buffers: sortedGpuBuffers,
      drawCount,
      instances: this.setupPrimitiveInstances(primitive, primitiveInstances),
      // No longer saving the material here!
    };

    const material = gltf.materials[primitive.material];
    const gpuMaterial = materialGpuData.get(material);

    const pipelineArgs = this.getPipelineArgs(primitive, sortedBufferLayout, material);
    const pipeline = this.getPipelineForPrimitive(pipelineArgs);

    // Rather than just storing a list of primitives for each pipeline store a map of
    // materials which use the pipeline to the primitives that use the material.
    let materialPrimitives = pipeline.materialPrimitives.get(gpuMaterial);
    if (!materialPrimitives) {
      materialPrimitives = [];
      pipeline.materialPrimitives.set(gpuMaterial, materialPrimitives);
    }

    // Add the primitive to the list of primitives for this material.
    materialPrimitives.push(gpuPrimitive);
}
```

### Updating the render loop for the new structure

Now to make use of this new structure we need to make on final, relatively small change to our render loop. Instead of iterating through the primitives for each pipeline, we're now going to iterate over the `materialPrimitives` entries, getting both the material and the list of associated primitives. We can then set the material's bind group once, then loop over all the primitves and render them as we did before.

```js
function renderGltf(renderPass) {
  renderPass.setBindGroup(0, frameBindGroup);
  renderPass.setBindGroup(1, instanceBindGroup);

  for (const gpuPipeline of pipelineGpuData.values()) {
    renderPass.setPipeline(gpuPipeline.pipeline);

    // Loop through every material that uses this pipeline and get an array of primitives
    // that uses that material.
    for (const [material, primitives] of gpuPipeline.materialPrimitives.entries()) {
      // Set the material bind group.
      renderPass.setBindGroup(2, material.bindGroup);

      // Loop through the primitives that use the current material/pipeline combo and draw
      // them as usual.
      for (const gpuPrimitive of primitives) {
        for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
          renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
        }

        renderPass.draw(gpuPrimitive.drawCount, gpuPrimitive.instances.count, 0, gpuPrimitive.instances.first);
      }
    }
  }
}
```

And just like that, we've now eliminated redundant material bind group setting! For a scene like "sponza" this takes us from 33 calls to `setBindGroup()` for the materials (the same number as the primitives) to 25 material `setBindGroup()` calls. That's a modest improvement, sure. As always, how effective this is will depend on how the scene is structured. But it's nice to know that your rendering isn't doing any more work than it has to, even when the models that it's being asked to display are less than ideal.

## That's a wrap!

And that, dear reader, brings us to the end of this (very long) case study in rendering glTF files with WebGPU. Now obviously we haven't got anything approaching a "fully featured" rendered at this point, but the patterns that we've covered in this document should hopefully lend themselves to being extended to handle more features, material properties, and edge cases.

## Appendix A: glTF/WebGPU helper functions

Some of the code snippets above reference utility functions for things like glTF enum translation to keep the example code simple, but the functions themselves are usually pretty straightforward! As such there's very little reason to not include them here. Feel free to use these as needed!

```js
function numberOfComponentsForType(type) {
  switch (type) {
    case 'SCALAR': return 1;
    case 'VEC2': return 2;
    case 'VEC3': return 3;
    case 'VEC4': return 4;
    default: return 0;
  }
}

function gpuFormatForAccessor(accessor) {
  const norm = accessor.normalized ? 'norm' : 'int';
  const count = numberOfComponentsForType(accessor.type);
  const x = count > 1 ? `x${count}` : '';
  switch (accessor.componentType) {
    case WebGLRenderingContext.BYTE: return `s${norm}8${x}`;
    case WebGLRenderingContext.UNSIGNED_BYTE: return `u${norm}8${x}`;
    case WebGLRenderingContext.SHORT: return `s${norm}16${x}`;
    case WebGLRenderingContext.UNSIGNED_SHORT: return `u${norm}16${x}`;
    case WebGLRenderingContext.UNSIGNED_INT: return `u${norm}32${x}`;
    case WebGLRenderingContext.FLOAT: return `float32${x}`;
  }
}

function gpuPrimitiveTopologyForMode(mode) {
  switch (mode) {
    case WebGLRenderingContext.TRIANGLES: return 'triangle-list';
    case WebGLRenderingContext.TRIANGLE_STRIP: return 'triangle-strip';
    case WebGLRenderingContext.LINES: return 'line-list';
    case WebGLRenderingContext.LINE_STRIP: return 'line-strip';
    case WebGLRenderingContext.POINTS: return 'point-list';
  }
}
```

## About the author

[Brandon Jones](https://twitter.com/tojiro) is a software engineer on the Google Chrome team and spec editor for WebGPU and WebXR.