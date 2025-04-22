import torch
import netron
import os
import argparse

from net.inceptionv3 import inception_v3, InceptionV3
from net.densenet_169_v2 import densenet169_v2
from net.vgg19 import vgg19_bn
from net.resnet101 import resnet101

def visualize_network(network_type="vgg19_bn", num_classes=2, input_shape=(1, 3, 320, 320), 
                      output_dir="./model_viz", model_name=None, open_viewer=True):
    """
    实例化网络并使用Netron生成可视化图表
    
    参数:
        network_type: 网络类型，可选 "vgg19" 或 "vgg19_bn"
        num_classes: 分类数量
        input_shape: 输入张量的形状，默认为(1, 3, 224, 224)
        output_dir: 输出目录
        model_name: 模型名称，如果为None，则使用network_type
        open_viewer: 是否打开Netron查看器
    
    返回:
        保存的图片路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果model_name为None，则使用network_type
    if model_name is None:
        model_name = network_type
        
    # 模型文件路径
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    image_path = os.path.join(output_dir, f"{model_name}.png")
    
    # 实例化网络
    print(f"正在实例化 {network_type} 网络...")
    if network_type == "inception_v3":
        model = inception_v3(
            num_classes=2,  # 保持为2，我们会在forward中选择第二个输出
            pretrained=False,
            input_size=320
        )
    elif model_name == "densenet169_v2":
        model = densenet169_v2(
            num_classes=2,  # 保持为2类输出
            pretrained=False,
            drop_rate=0.0,
            memory_efficient=True,
            img_size=320
        )
    elif model_name == "vgg19_bn":
        model = vgg19_bn(
            num_classes=2,
            dropout=0.0,
            init_weights=True
        )
    elif model_name == "resnet101":  # 添加这个条件分支
        model = resnet101(
            num_classes=2,
            dropout=0.0,
            init_weights=True
        )
    else:
        raise ValueError(f"不支持的网络类型: {network_type}")
    
    # 设置为评估模式
    model.eval()
    
    # 创建一个示例输入
    dummy_input = torch.randn(input_shape)
    
    try:
        # 尝试导出为ONNX格式，使用较高的opset版本
        print(f"正在将模型导出为ONNX格式...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,  # 尝试使用更高的opset版本
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        print(f"ONNX模型已保存至 {onnx_path}")
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        print("\n正在尝试替代方法...")
        
        # 替代方法: 使用PyTorch的JIT跟踪并保存为PT文件
        pt_path = os.path.join(output_dir, f"{model_name}.pt")
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, pt_path)
        print(f"已保存为PyTorch JIT模型: {pt_path}")
        
        # 使用netron查看PT文件
        onnx_path = pt_path
    
    # 使用Netron保存网络结构图
    if open_viewer:
        # 打开Netron查看器
        print(f"正在打开Netron查看器，请在查看器中手动保存图片")
        netron.start(onnx_path, browse=True)
    else:
        # 直接保存图片
        print(f"正在保存网络结构图到 {image_path}")
        try:
            netron.export_to_png(onnx_path, image_path)
        except:
            print(f"无法自动导出图片，请手动打开 {onnx_path} 查看模型结构")
    
    return onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch网络结构可视化工具")
    parser.add_argument("--type", type=str, default="vgg19_bn", choices=["vgg19_bn", "inception_v3", "densenet169_v2", "resnet101"], 
                       help="网络类型，可选 vgg19 或 vgg19_bn")
    parser.add_argument("--classes", type=int, default=2, help="分类数量")
    parser.add_argument("--shape", type=int, nargs='+', default=[1, 3, 320, 320], 
                       help="输入形状，默认为[1, 3, 224, 224]")
    parser.add_argument("--output", type=str, default="./model_viz", help="输出目录")
    parser.add_argument("--name", type=str, default=None, help="模型名称")
    parser.add_argument("--no-viewer", default=True, action="store_true", help="不打开Netron查看器，只保存图片")
    
    args = parser.parse_args()
    
    # 可视化网络
    visualize_network(
        network_type=args.type,
        num_classes=args.classes,
        input_shape=tuple(args.shape),
        output_dir=args.output,
        model_name=args.name,
        open_viewer=not args.no_viewer
    )
