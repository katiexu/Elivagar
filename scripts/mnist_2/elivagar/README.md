| 脚本 | 作用 | 快照信息的使用 | 实际模拟方式 | 调用 FakeTorino 后端？ |
| --- | --- | --- | --- | --- |
| circ_gen_osaka.sh | 生成候选电路和物理比特映射 | 直接读取硬件配置和校准信息 | 不运行电路模拟 | 否 |
| cnr_osaka.sh | 计算电路的噪声鲁棒性 CNR | 直接读取快照中的噪声模型、门集和连接关系 | 用 AerSimulator 分别进行有噪声和无噪声模拟 | 否 |
| repcap_osaka.sh | 计算电路表达能力 RepCap | 不读取快照；使用已生成的电路 | 用 torchquantum 进行无噪声模拟 | 否 |
| train_osaka.sh | 根据评分筛选电路并训练 | 不读取快照；筛选时使用基于快照算出的 CNR | 用 torchquantum 训练，无 FakeTorino 噪声模型 | 否 |
| eval_osaka.sh | 评估训练后模型的噪声下准确率 | 直接读取快照中的噪声模型和门集等信息 | 用 AerSimulator 进行有噪声推理 | 否 |
