## 人工智能monorepo单体式仓库 单仓多包

#### 项目介绍

- 项目代号: athena(雅典娜 智慧女神) 愿景: 使项目更易于复用迭代维护扩展、分离关注点并避免代码重复

#### 机器学习技术栈

- PyTorch
- TensorFlow
- JAX

#### 模块目录结构

- athena-common: 公共通用模块 (与业务无关，大部分模块项目依赖需要)
- athena-dao  : 数据库通用配置层模块
- athena-util  : 工具模块
- athena-constant  : 项目常量 枚举和通用yaml配置模块
- athena-test : 测试 、实验性功能等模块 (与生产环境无关 可随便折腾😅)
- athena-generator : 自动代码生成等模块
- business-common-service : 业务通用服务模块组模块

#### Git Commit Message格式说明一般包括三部分：Header、Body和Footer

message信息格式采用目前主流的Angular规范，是目前使用最广的写法，比较合理和系统化，并且有配套的工具 在IDEA 可安装Git Commit Message插件 自动生成规范提交信息

##### Header

type(scope): subject

type: 用于说明commit的类别，规定为如下几种

- feat: 新增功能
- fix: 修复 bug
- docs: 修改文档 比如 README, CHANGELOG, CONTRIBUTE等
- refactor: 代码重构，未新增任何功能和修复任何 bug
- build: 改变构建流程，新增依赖库、工具等（例如 webpack，maven等修改）
- style: 仅仅修改了空格、缩进、逗号等，不改变代码逻辑
- perf: 改善性能和体验的修改
- chore: 非源码和测试文件修改的杂项处理
- test: 测试用例的修改 包括单元测试、集成测试等
- ci: 持续集成相关
- release: 自动化触发CI/CD流水线
- revert: 回滚到上一个版本
- scope: 【可选】用于说明 commit 的影响范围
- subject: commit 的简要说明，尽量简短

##### Body

对本次commit的详细描述，可分多行

##### Footer

不兼容变动：需要描述相关信息 关闭指定Issue：输入Issue信息

##### 示例 英文:冒号后加单空格 再加提交说明

feat(app): 用户登录功能

#### IDEA工具内File->Settings->File and Code Templates->Includes标签->File Header 添加如下注释信息，之后创建类会自动生成注释信息

/**
* @author 潘维吉
* @date ${DATE} ${TIME}
* @description 描述
*/