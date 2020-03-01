use winit::{
    event_loop::{EventLoop, ControlFlow},
    event::*,
    window::Window,
    dpi::PhysicalSize,
};
use image::GenericImageView;
use cgmath::SquareMatrix;
use std::time::{Instant, Duration};

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let time = Instant::now() + Duration::from_millis(40);

    let id = window.id();

    let mut cube = Cube::new(&window);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == id => if cube.input(event) {
                *control_flow = ControlFlow::WaitUntil(time);
            } else {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input: i,
                        ..
                    } => match i {
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        _ => *control_flow = ControlFlow::WaitUntil(time),
                    },
                    WindowEvent::Resized(size) => {
                        cube.resize(*size);
                        *control_flow = ControlFlow::WaitUntil(time);
                    },
                    WindowEvent::ScaleFactorChanged {
                        new_inner_size,
                        ..
                    } => {
                        cube.resize(**new_inner_size);
                        *control_flow = ControlFlow::WaitUntil(time);
                    },
                    _ => *control_flow = ControlFlow::WaitUntil(time),
                }
            }
            Event::MainEventsCleared => {
                cube.update();
                cube.render();
                *control_flow = ControlFlow::WaitUntil(time);
            },
            _ => *control_flow = ControlFlow::WaitUntil(time),
        }
    });
}

#[allow(dead_code)]
struct Cube {
    adapter: wgpu::Adapter,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain: wgpu::SwapChain,
    sc_desc: wgpu::SwapChainDescriptor,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    diffuse_texture: wgpu::Texture,
    diffuse_texture_view: wgpu::TextureView,
    diffuse_sampler: wgpu::Sampler,
    diffuse_bind_group: wgpu::BindGroup,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl Cube {
    fn new(window: &Window) -> Self {
        let inner_size = window.inner_size();
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            ..Default::default()
        }).unwrap();
        let surface = wgpu::Surface::create(window);

        let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        });

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: inner_size.width,
            height: inner_size.height,
            present_mode: wgpu::PresentMode::Vsync,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let light = Light {
            direction: (0.9, 0.4, 1.0).into(),
        };

        let light_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM).fill_from_slice(&[light]);

        let camera = Camera {
            eye: (3.0, 3.0, 3.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: sc_desc.width as f32 / sc_desc.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            theta: 0.0,
        };

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST).fill_from_slice(&[uniforms]);

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            bindings: &[
                wgpu::BindGroupLayoutBinding{
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                    },
                },
            ],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            layout: &uniform_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buffer,
                        range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &light_buffer,
                        range: 0..std::mem::size_of_val(&light) as wgpu::BufferAddress,
                    },
                },
            ],
        });

        let diffuse_bytes = include_bytes!("abcd.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.as_rgba8().unwrap();
        
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size.clone(),
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        let diffuse_buffer = device.create_buffer_mapped(diffuse_rgba.len(), wgpu::BufferUsage::COPY_SRC).fill_from_slice(&diffuse_rgba);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &diffuse_buffer,
                offset: 0,
                row_pitch: 4 * dimensions.0,
                image_height: dimensions.1,
            },
            wgpu::TextureCopyView {
                texture: &diffuse_texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            texture_size,
        );

        queue.submit(&[encoder.finish()]);

        let diffuse_texture_view = diffuse_texture.create_default_view();

        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor{
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
            mipmap_filter: wgpu::FilterMode::Linear,
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            bindings: &[
                wgpu::BindGroupLayoutBinding{
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler,
                },
            ],
        });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
        });

        let vertex_buffer = device.create_buffer_mapped(VERTICES.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(VERTICES);

        let index_buffer = device.create_buffer_mapped(INDICES.len(), wgpu::BufferUsage::INDEX).fill_from_slice(INDICES);

        let num_indices = INDICES.len() as u32;

        let vs_src = include_str!("shader.vert");
        let fs_src = include_str!("shader.frag");

        let vs_spriv = glsl_to_spirv::compile(vs_src, glsl_to_spirv::ShaderType::Vertex).unwrap();
        let fs_spirv = glsl_to_spirv::compile(fs_src, glsl_to_spirv::ShaderType::Fragment).unwrap();

        let vs_data = wgpu::read_spirv(vs_spriv).unwrap();
        let fs_data = wgpu::read_spirv(fs_spirv).unwrap();

        let vs_module = device.create_shader_module(&vs_data);
        let fs_module = device.create_shader_module(&fs_data);

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            layout: &render_pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor{
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor{
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[
                Vertex::desc(),
            ],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            adapter,
            device,
            queue,
            sc_desc,
            swap_chain,
            surface,
            vertex_buffer,
            index_buffer,
            num_indices,
            render_pipeline,
            camera,
            diffuse_sampler,
            diffuse_texture,
            diffuse_texture_view,
            diffuse_bind_group,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    fn render(&mut self) {
        let frame = self.swap_chain.get_next_texture();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            todo: 0,
        });

        {    
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        },
                    }
                ],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffers(0, &[(&self.vertex_buffer, 0)]);
            render_pass.set_index_buffer(&self.index_buffer, 0);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(&[
            encoder.finish()
        ]);
    }

    fn update(&mut self) {
        let r:f32 = 2.0f32.sqrt() * 3.0f32;
        self.camera.theta += 0.005;
        if self.camera.theta > 3.14 {
            self.camera.theta = -3.14;
        }
        let sint = self.camera.theta.sin();
        let cost = self.camera.theta.cos();
        let (x, y) = ( (r * cost) as f32, (r * sint) as f32 );
        self.camera.eye = ( x, y, 3.0).into();

        self.uniforms.update_view_proj(&self.camera);

        let temp_buf = self.device.create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC).fill_from_slice(&[self.uniforms]);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            todo: 0,
        });

        encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.uniform_buffer, 0, std::mem::size_of::<Uniforms>() as wgpu::BufferAddress);

        self.queue.submit(&[encoder.finish()]);
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

const VERTICES: &[Vertex] = &[
    //Right
    Vertex { position: [1.0, 1.0, -1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, -1.0, 1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, -1.0, -1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
    //Left
    Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
    //Front
    Vertex { position: [1.0, -1.0, -1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, -1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
    //Back
    Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, -1.0, 1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, 1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
    //Top
    Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, -1.0, -1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, -1.0, 1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
    //Bottom
    Vertex { position: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, -1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },
];

const INDICES: &[u16] = &[
    //Right
    0, 1, 3,
    3, 1, 2,
    //Left
    5, 4, 7,
    7, 6, 5,
    //Front
    8, 9, 10,
    10, 11, 8,
    //Back
    12, 13, 15,
    15, 13, 14,
    //Top
    16, 17, 18,
    18, 19, 16,
    //Bottom
    20, 21, 23,
    23, 21, 22,
];

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0u64,
                    format: wgpu::VertexFormat::Float3,
                    shader_location: 0u32,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float2,
                    shader_location: 1u32, 
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float3,
                    shader_location: 2u32,
                },
            ],
        }
    }
}

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    theta: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Uniforms {
    view_proj: cgmath::Matrix4<f32>,
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix();
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Light {
    direction: cgmath::Vector3<f32>,
}