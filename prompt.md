我需要如下功能：
- 监控一个指定文件,如果文件被修改就触发下列内容(不允许轮询,需要被文件修改触发)
    - 读取文件内容
    - 将文件内容作为字符串送入函数apply_template(file_content: string) -> string
    - 将返回的内容保存到一个临时文件中
    - 在指定的PYTHONPATH变量下用python执行这个文件
    - 捕获stdout和stderr的所有内容为字符串
    - 将捕获的内容送入parse_output(captured: string) -> string
    - 打印输出的内容,我希望每次终端都清空.
我应该怎么实现? python有好的实现方式吗?
启动方式只指定目标文件路径作为唯一cli arg