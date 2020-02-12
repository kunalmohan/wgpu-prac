use winit::{
    event_loop::{EventLoop, ControlFlow},
    event::*,
    window::Window,
    dpi::PhysicalSize,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let mut cube = Cube::new(window);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if cube.input(event) {
                *control_flow = ControlFlow::Wait;
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
                        _ => *control_flow = ControlFlow::Wait,
                    },
                    WindowEvent::Resized(size) => {
                        cube.resize(*size);
                        *control_flow = ControlFlow::Wait;
                    },
                    WindowEvent::ScaleFactorChanged {
                        new_inner_size,
                        ..
                    } => {
                        cube.resize(**new_inner_size);
                        *control_flow = ControlFlow::Wait;
                    },
                    _ => *control_flow = ControlFlow::Wait,
                },
            }
            Event::MainEventsCleared => {
                cube.update();
                cube.render();
                *control_flow = ControlFlow::Wait;
            }
            - => *control_flow = ControlFlow::Wait;
        }
    });
}

struct Cube {
    adapter: wgpu::Adapter,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain: wgpu::SwapChain,
    sc_desc: wgpu::SwapChainDescriptor,
  //  buffer: wgpu::Buffer,
}

impl Cube {
    fn new(window: Window) -> Self {
        let inner_size = window.inner_size();
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            ..Default::default()
        }).unwrap();
        let surface = wgpu::Surface::create(&window);

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        });

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Snorm,
            width: inner_size.width,
            height: inner_size.height,
            present_mode: wgpu::PresentMode::Vsync,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        /*let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: ,
            usage: wgpu::BufferUsage::
        })*/

        Self {
            adapter,
            device,
            queue,
            sc_desc,
            swap_chain,
            surface,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    fn render(&self) {

    }

    fn update(&self) {

    }

    fn input(&self) {
        false
    }
}