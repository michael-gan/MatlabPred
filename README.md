**开启展示页面：**

markdown版: 直接打开文件`MATLAB Predictive Maintenance Toolbox Introduction.md`

pdf版：直接打开文件`MATLAB Predictive Maintenance Toolbox Introduction.pdf`

网页slides版：打开`index.html`查看案例，打开`工具箱函数介绍.html`查看工具箱函数接口

**Tips for author：**

`pandoc`转换`.md`文件为`revealjs`支持的slides网页

1. 使用命令`pandoc demo.md -o demo.html -t revealjs -s -V theme=simple`，其中`demo.md`表示想要转换的原始`markdown`文件，`deml.html`表示目标文件名，`theme=simple`表示主题，具体可选主题可参考`reveal.js`安装文件夹中的`css/theme`部分。
2. 手动将生成的`.html`文件中的`revealjs`文件目录改成安装的绝对路径，如从`reveal.js/css/theme/moon.css`改为`C:/Users/jian.gan/reveal.js/css/theme/moon.css`。生成的`.html`文件以`.md`文件中的标题级别作为slide的划分依据。
