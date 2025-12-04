import sys
import os
import tempfile
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def apply_template(file_content: str) -> str:
    return file_content


def parse_output(captured: str) -> str:
    return captured


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, target_file: Path):
        self.target_file = target_file.resolve()

    def on_modified(self, event):
        # 只处理目标文件的修改事件
        if event.is_directory:
            return

        modified_file = Path(event.src_path).resolve()
        if modified_file != self.target_file:
            return

        self.process_file()

    def process_file(self):
        """处理文件修改事件的主逻辑"""
        try:
            # 1. 读取文件内容
            with open(self.target_file, "r", encoding="utf-8") as f:
                file_content = f.read()

            # 2. 应用模板
            processed_content = apply_template(file_content)

            # 3. 保存到临时文件
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(processed_content)
                tmp_file_path = tmp_file.name

            try:
                # 4. 设置环境变量并执行
                env = os.environ.copy()

                result = subprocess.run(
                    [sys.executable, tmp_file_path],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30,  # 设置超时防止无限运行
                )

                # 5. 捕获输出
                captured = result.stdout
                if result.stderr:
                    captured += "\n" + result.stderr

                # 6. 解析输出
                parsed_output = parse_output(captured)

                # 7. 清空终端并打印
                os.system("cls" if os.name == "nt" else "clear")
                print(parsed_output)

                # 显示返回码
                if result.returncode != 0:
                    print(f"\n[警告] 进程退出码: {result.returncode}")

            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass

        except Exception as e:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"[错误] 处理文件时出错: {e}")
            import traceback

            traceback.print_exc()


def main():
    if len(sys.argv) != 2:
        print("使用方法: python watch_and_run.py <target_file>")
        sys.exit(1)

    target_file = Path(sys.argv[1])

    if not target_file.exists():
        print(f"错误: 文件不存在: {target_file}")
        sys.exit(1)

    if not target_file.is_file():
        print(f"错误: 不是一个文件: {target_file}")
        sys.exit(1)

    print("等待文件修改...")
    print("-" * 50)

    # 创建事件处理器和观察者
    event_handler = FileChangeHandler(target_file)
    observer = Observer()

    # 监控文件所在目录
    watch_directory = target_file.parent
    observer.schedule(event_handler, str(watch_directory), recursive=False)

    # 启动监控
    observer.start()

    try:
        # 保持运行
        observer.join()
    except KeyboardInterrupt:
        print("\n\n停止监控...")
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
