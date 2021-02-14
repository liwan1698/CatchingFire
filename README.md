# CatchingFire
文本标注工具，包括实体识别标注、文本分类标注、三元组抽取标注。支持规则、机器学习模型、深度学习模型辅助标注。<br>
文本分类模型采用fasttext，实体识别采用bert+bilstm+crf，三元组抽取采用bert。<br>
后端web框架使用django，前端使用vue。

# 架构
![架构图](https://raw.githubusercontent.com/deepwel/Chinese-Annotator/master/docs/images/chinese_annotator_arch.png) <br>
注：参考https://github.com/crownpku/Chinese-Annotator

# 目录结构描述
```
.
├── backend      // 后端django
│   └── algo  // 算法模块
│   └── backend  // 后端设置
│   └── data  // 数据样例
│   └── save_model  // 保存的模型
│   └── db.sqlite3  // sqlite数据库
│   └── process_data_model.py  // 数据预处理
├── data      // 数据
├── test      // 测试代码
```

# 合作
现在只有我一个人开发，因此需要前端和算法的小伙伴合作开发，有兴趣可以加VX（li805174247）询问。

# todo list
```
前端页面
支持配置规则
配置模块

```
