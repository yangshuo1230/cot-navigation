# 6/9层纠错attention对照

入口：[comparison.html](comparison.html)；完整解释：[REPORT.md](REPORT.md)。本地服务若已启动，可打开 `http://127.0.0.1:8765/correction_attention_compare_20261102/comparison.html`。

五个冻结checkpoint：原6层185k；30%错误前缀训练的6/9层、seed0/1。相同18val+18test正确参考query，原始attention，不重新训练模型。checkpoint身份见models.json。

服务器项目根目录`.venv/bin/python studies/correction_attention_compare_20261102/<脚本>`，CPU4线程：collect.py → summarize.py → supplement.py → extra_summary.py → intervene.py/intervene_extra.py/bar_pool.py/graph_writer.py/wrong_visited.py。本地summary/extra_summary/build_report用`D:\anaconda\python.exe`，生成PNG、内嵌图片HTML及报告验证。

对照不是新benchmark成绩：候选合法顺序不唯一，主统计teacher-forced；实际局部字段生成及人工错误条件单独报告。不同best步数、seed和训练配方的混杂限制见报告。
