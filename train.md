train():
1. 初始化
   └─ 设置 model、model_teacher、model_ta 为训练模式，准备数据加载器迭代器等。
2. 训练循环
   ├─ 2.1 准备输入数据
   │    ├─ 加载带标签数据：image_l 和 label_l（原始标签）
   │    └─ 加载无标签数据：image_u
   ├─ 2.2 生成伪标签
   │    └─ 使用 model_teacher 生成无标签数据的伪标签：label_u_aug
   ├─ 2.3 强数据增强
   │    └─ 对无标签数据 image_u 应用强数据增强策略
   ├─ 2.4 前向传播
   │    ├─ 将 image_l 和 image_u 输入到 model(虽然model见过image_l和image_u,但计算损失的时候只用image_l)
   │    ├─ 将 image_l 和 image_u 输入到 model_ta(虽然model_ta见过image_l和image_u,但计算损失的时候只用image_u)
   │    └─ 将 image_l 和 image_u 输入到 model_teacher(虽然model_ta见过image_l和image_u,但其参数是来自于model的EMA更新得到的)
   ├─ 2.5 计算损失
   │    ├─ 计算有监督损失（带标签数据：image_l 和 label_l）
   │    │    ├─ 使用损失函数：sup_loss_fn
   │    │    ├─ 输入数据：pred_l_large 和 label_l（原始标签）
   │    │    └─ 如果使用辅助损失，则输入数据：pred_l_large、aux 和 label_l（原始标签）
   │    └─ 计算无监督损失（无标签数据：image_u）
   │         ├─ 使用损失函数：compute_unsupervised_loss_conf_weight
   │         ├─ 输入数据：pred_a_u_large（来自 model_ta）、label_u_aug（来自 model_teacher）、pred_u_large_teacher（来自 model_teacher）
   │         └─ 损失权重：cfg["trainer"]["unsupervised"].get("loss_weight", 1)
   ├─ 2.6 反向传播与参数更新
   │    ├─ 根据有监督损失，更新 model 的参数,使用指数滑动平均（EMA）策略更新 ,model 的参数（基于 model_ta 的参数）
   │    └─ 根据无监督损失，更新 model_ta 的参数
   ├─ 2.7 更新教师模型参数
   │    └─ 使用指数滑动平均（EMA）策略更新 model_teacher 的参数（基于 model 的参数）
   └─ 2.8 记录性能指标
       └─ 更新损失、批处理时间等指标并记录到 TensorBoard
3. 结束训练循环


