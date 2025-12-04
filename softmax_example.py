#!/usr/bin/env python3
"""
这个脚本演示如何在不调用 CUDA C 库的情况下编译 cuda.tile kernel 并 dump IR。
适用于 CUDA driver 版本不足的情况。

使用方法：
    python example.py

生成的 IR 文件将保存在 ./ir_artifacts 目录下：
    - softmax.cutileir: CuTile IR (高级 IR)
    - softmax.cutile: Bytecode (序列化的 IR)
    - softmax.cuda_tile.mlir: MLIR (如果 cuda.tile_internal 可用)
"""

import os
import sys
import functools

# 设置环境变量以 dump IR
os.environ["CUDA_TILE_DUMP_TILEIR"] = "./ir_artifacts"
os.environ["CUDA_TILE_DUMP_BYTECODE"] = "./ir_artifacts"

# 现在可以正常导入 cuda.tile 了（因为我们创建了 mock _cext.py）
import cuda.tile as ct
from cuda.tile._compile import _get_final_ir
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._cext import default_tile_context, get_compute_capability
from cuda.tile._ir import ir
import cuda.tile._bytecode as bc
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel


# 定义 kernel 函数
@ct.kernel
def softmax(x, out, r: ct.Constant):
    # r 被标记为 Constant，所以可以在 load 的 shape 参数中使用
    c = ct.load(x, (0, ct.bid(0)), (512, r))
    max = ct.max(c, axis=0, keepdims=True)
    num = ct.exp(c - max)
    den = ct.sum(num, axis=0, keepdims=True)
    smax = num / den
    ct.store(out, (0, ct.bid(0)), smax)


# 创建模拟的 tensor 对象用于类型推断（不需要真实的 CUDA 设备）
class MockTensor:
    def __init__(self, shape, dtype_str="float32"):
        self.shape = shape
        self.dtype_str = dtype_str
        self.device = "cuda"

        # 模拟 torch.dtype
        class MockDtype:
            def __init__(self, name):
                self.name = name

        self.dtype = MockDtype(dtype_str)
        self.data_ptr = lambda: 0  # 模拟指针

        # 实现 __cuda_array_interface__ 协议
        # 这是 CUDA 数组的标准接口
        dtype_map = {
            "float32": "<f4",
            "float64": "<f8",
            "float16": "<f2",
            "int32": "<i4",
            "int64": "<i8",
            "int16": "<i2",
            "int8": "<i1",
            "uint32": "<u4",
            "uint64": "<u8",
            "uint16": "<u2",
            "uint8": "<u1",
        }

        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": dtype_map.get(dtype_str, "<f4"),
            "data": (0, False),  # (ptr, read_only)
            "version": 3,
        }


def compile_and_dump_ir():
    """编译 kernel 并 dump IR"""

    # 使用 mock tensor 进行编译
    x = MockTensor((512, 128), "float32")
    out = MockTensor((512, 128), "float32")
    r = 8

    pyfunc = softmax._pyfunc  # 获取原始 Python 函数
    args = (x, out, r)

    # 创建编译选项
    compiler_options = CompilerOptions(num_ctas=None, occupancy=None, opt_level=3)

    # 获取 sm_arch（使用 mock 的 get_compute_capability）
    major, minor = get_compute_capability()
    sm_arch = f"sm_{major}{minor}"
    print(f"目标架构: {sm_arch}", file=sys.stderr)

    try:
        # 获取 IR
        print("正在生成 CuTile IR...", file=sys.stderr)
        func_ir = _get_final_ir(pyfunc, args, default_tile_context)

        # 打印 CuTile IR
        print("\n" + "=" * 60, file=sys.stderr)
        print("CuTile IR", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        ir_string = func_ir.to_string(include_loc=False)
        print(ir_string, file=sys.stderr)

        # 生成 bytecode
        print("\n正在生成 bytecode...", file=sys.stderr)
        bytecode_generator = functools.partial(
            generate_bytecode_for_kernel, func_ir, compiler_options, sm_arch
        )

        bytecode_buf = bytearray()
        with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
            bytecode_generator(writer, anonymize_debug_attr=False)

        # 保存 bytecode
        os.makedirs("./ir_artifacts", exist_ok=True)
        bytecode_path = "./ir_artifacts/softmax.cutile"
        with open(bytecode_path, "wb") as f:
            f.write(bytecode_buf)
        print(f"✓ Bytecode 已保存到: {bytecode_path}", file=sys.stderr)

        # 保存 CuTile IR 到文件
        ir_path = "./ir_artifacts/softmax.cutileir"
        with open(ir_path, "w") as f:
            f.write(ir_string)
        print(f"✓ CuTile IR 已保存到: {ir_path}", file=sys.stderr)

        # 尝试转换为 MLIR 并打印
        try:
            from cuda.tile_internal._internal_cext import bytecode_to_mlir_text

            mlir_text = bytecode_to_mlir_text(bytecode_buf)
            print("\n" + "=" * 60, file=sys.stderr)
            print("TILEIR MLIR", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print(mlir_text, file=sys.stderr)

            mlir_path = "./ir_artifacts/softmax.cuda_tile.mlir"
            with open(mlir_path, "w") as f:
                f.write(mlir_text)
            print(f"✓ MLIR 已保存到: {mlir_path}", file=sys.stderr)
        except ImportError:
            print("\n注意: 无法导入 cuda.tile_internal，跳过 MLIR 转换", file=sys.stderr)
            print("      这是正常的，因为 MLIR 转换需要额外的内部扩展", file=sys.stderr)

        print("\n" + "=" * 60, file=sys.stderr)
        print("✓ 编译成功！IR 已 dump 到 ./ir_artifacts 目录", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        return True

    except Exception as e:
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"✗ 编译失败: {e}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = compile_and_dump_ir()
    sys.exit(0 if success else 1)
