---
title: Intellij IDEA插件开发入门（一）
date: 2019-01-27 21:13:10
tags:
    - Intellij Plugin
categories:
    - Notes
photos:
    - 0.png
---

Intellij IDEA插件开发有两种方式：

* Gradle
* Plugin Devkit

本文根据官方推荐使用Gradle。

## 1. 插件开发环境

* IDEA: 社区版本
* Project JDK: 1.8
* Gradle: 4.10

<!-- more -->

## 2. 确认Gradle可用

菜单Preferences -> Plugins

![1](Plugin1/1.png)

## 3. 创建Plugin项目

![2](Plugin1/2.png)

![3](Plugin1/3.png)

![4](Plugin1/4.png)

（官方推荐勾选“Use default cradle wrapper”，以便IDEA自动安装Gradle需要的包）

![5](Plugin1/5.png)

项目创建完成。

**工程结构：**

![](Plugin1/16.png)

![6](Plugin1/17.png)

**plugin.xml文件内容：**

* id：当前插件的唯一id号。
* name：插件的名称。
* version：插件的版本号。
* vendor：开发人的邮箱、公司名称。
* description：插件的描述，如果将插件上传到IDEA的仓库，在进行下载时会显示该描述。
* idea-version：表示当前插件所支持的所有IDEA版本。
* extensions：一般放一些我们自己扩展的东西，比如新增高亮显示、新增语言支持。
* actions：新增的类在这里注册，用于菜单栏扩展。

## 4. 配置Gradle插件

在build.gradle文件中，设置运行插件的沙箱地址。

![7](Plugin1/7.png)

## 5. 创建一个action

![8](Plugin1/8.png)

![9](Plugin1/9.png)

自定义功能加在Window菜单栏下。

![10](Plugin1/10.png)

![11](Plugin1/11.png)

在plugin.xml文件中，项目自动生成action配置：

![12](Plugin1/12.png)

## 6. Gradle运行配置

菜单Edit Configurations -> Run/Debug Configurations

点击'+'号，新建Gradle Run Configuration。

![13](Plugin1/18.png)



![14](Plugin1/19.png)

![15](Plugin1/13.png)

## 7. 运行项目

![16](Plugin1/20.png)

在Window菜单栏加入我们自定义的'Greeting'选项，点击弹出'Hello World!'。

![17](Plugin1/14.png)

![18](Plugin1/15.png)

## 8. 打包插件

参考文献：

IDEA官方插件开发手册http://www.jetbrains.org/intellij/sdk/docs/basics.html

