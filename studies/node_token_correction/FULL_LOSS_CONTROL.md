# 错误token计入loss对照

新增12个repair_full_loss模型，与repair逐seed、协议、图家族配对，总计36个模型。
两组训练token序列完全一致，仅错误token的监督mask不同；MISTAKE、正确节点及后缀始终受监督。复用同一自然错误池、曝光随机流、272-token词表及1M步九级课程。

已验证12对配置的data/model/train完全一致，protocol仅mistake_mask_errors不同；构造样本验证了两组输入token相同，监督位置集合仅多出错误token。

新后台调度PID3157782，组名node_token_full_loss_20261029；等待原24任务完成后自动启动，以避免两个调度器竞争GPU。原训练未中断。模型结果仍保存在原组目录对应repair_full_loss子目录，新调度状态保存在outputs/node_token_full_loss_20261029。
