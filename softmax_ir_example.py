#!/usr/bin/env python3
"""
Softmax Kernel IR Dump 示例

使用 ir_dump 库编译 softmax kernel 并导出 IR

使用方法：
    python softmax_ir_example.py

生成的 IR 文件将保存在 ./ir_artifacts 目录下：
    - softmax.cutileir: CuTile IR (高级 IR)
    - softmax.cutile: Bytecode (序列化的 IR)
"""

import sys
import cuda.tile as ct
from ir_dump import CutileIrDump


# 定义 softmax kernel
@ct.kernel
def softmax(x, out, r: ct.Constant):
    # r 被标记为 Constant，所以可以在 load 的 shape 参数中使用
    c = ct.load(x, (0, ct.bid(0)), (512, r))
    max = ct.max(c, axis=0, keepdims=True)
    num = ct.exp(c - max)
    den = ct.sum(num, axis=0, keepdims=True)
    smax = num / den
    ct.store(out, (0, ct.bid(0)), smax)


def main():
    """主函数"""
    print("=" * 60)
    print("Softmax Kernel IR Dump 示例")
    print("=" * 60)

    # 创建 IR dumper
    dumper = CutileIrDump(
        output_dir="./ir_artifacts",
        dump_cutileir=True,
        dump_bytecode=True,
        dump_mlir=False,
    )

    # 创建 mock tensors
    x = dumper.create_mock_tensor((512, 128), dtype="float32")
    out = dumper.create_mock_tensor((512, 128), dtype="float32")
    r = 8

    # 编译并导出 IR
    print("\n正在编译 softmax kernel...")
    try:
        files = dumper.compile_kernel(softmax, args=[x, out, r], kernel_name="softmax")

        print("\n✓ 编译成功！生成的文件：")
        for ir_type, path in files.items():
            print(f"  - {ir_type}: {path}")

        # 打印 IR 内容预览
        print("\n" + "=" * 60)
        print("CuTile IR 预览（前 20 行）")
        print("=" * 60)
        ir_string = dumper.dump_ir_to_string(softmax, args=[x, out, r], ir_type="cutileir")
        lines = ir_string.split("\n")[:20]
        print("\n".join(lines))
        if len(ir_string.split("\n")) > 20:
            print("...")

        print("\n" + "=" * 60)
        print("✓ 完成！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 编译失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
