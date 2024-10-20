[合集 \- Ubuntu强化学习合集(5\)](https://github.com)[1\.ROS基础入门——实操教程10\-04](https://github.com/hassle/p/18447212)[2\.强化学习算法笔记之【Q\-learning算法和DQN算法】10\-18](https://github.com/hassle/p/18473878)3\.强化学习算法笔记之【DDPG算法】10\-19[4\.强化学习笔记之【SAC算法】10\-11](https://github.com/hassle/p/18459320)[5\.强化学习笔记之【论文精读】【ACE:一种基于熵规整和因果关系的离线SAC算法】10\-17](https://github.com/hassle/p/18472441)收起
# 强化学习笔记之【DDPG算法】


目录* [强化学习笔记之【DDPG算法】](https://github.com)
	+ - [前言：](https://github.com)
		- [原论文伪代码](https://github.com)
		- [DDPG 中的四个网络](https://github.com):[蓝猫机场加速器](https://dahelaoshi.com)
		- [代码核心更新公式](https://github.com)



---


### 前言：


本文为强化学习笔记第二篇，第一篇讲的是Q\-learning和DQN


就是因为DDPG引入了Actor\-Critic模型，所以比DQN多了两个网络，网络名字功能变了一下，其它的就是软更新之类的小改动而已


本文初编辑于2024\.10\.6


CSDN主页：[https://blog.csdn.net/rvdgdsva](https://github.com)


博客园主页：[https://github.com/hassle](https://github.com)


博客园本文链接：


![](https://img2024.cnblogs.com/blog/3382553/202410/3382553-20241018111324199-720912544.jpg)
真 · 图文无关




---


### 原论文伪代码


![](https://img2024.cnblogs.com/blog/3382553/202410/3382553-20241018111324670-465692421.png)
* 上述代码为DDPG原论文中的伪代码




---


需要先看：


[Deep Reinforcement Learning (DRL) 算法在 PyTorch 中的实现与应用](https://github.com)【DDPG部分】【没有在选择一个新的动作的时候，给policy函数返回的动作值增加一个噪音】【critic网络与下面不同】


[深度强化学习笔记——DDPG原理及实现（pytorch）](https://github.com)【DDPG伪代码部分】【这个跟上面的一样没有加噪音】【critic网络与上面不同】


[【深度强化学习】(4\) Actor\-Critic 模型解析，附Pytorch完整代码](https://github.com)【选看】【Actor\-Critic理论部分】




---


如果需要给policy函数返回的动作值增加一个噪音，实现如下


![](https://img2024.cnblogs.com/blog/3382553/202410/3382553-20241018111325059-726698426.png)

```
def select_action(self, state, noise_std=0.1):
    state = torch.FloatTensor(state.reshape(1, -1))
    action = self.actor(state).cpu().data.numpy().flatten()
    
    # 添加噪音，上面两个文档的代码都没有这个步骤
    noise = np.random.normal(0, noise_std, size=action.shape)
    action = action + noise
    
    return action


```



---


### DDPG 中的四个网络


![](https://img2024.cnblogs.com/blog/3382553/202410/3382553-20241018111325416-62119828.png)
**注意！！！这个图只展示了Critic网络的更新，没有展示Actor网络的更新**


* **Actor 网络（策略网络）**：
	+ **作用**：决定给定状态 ss 时，应该采取的动作 a\=π(s)a\=π(s)，目标是找到最大化未来回报的策略。
	+ **更新**：基于 Critic 网络提供的 Q 值更新，以最大化 Critic 估计的 Q 值。
* **Target Actor 网络（目标策略网络）**：
	+ **作用**：为 Critic 网络提供更新目标，目的是让目标 Q 值的更新更为稳定。
	+ **更新**：使用软更新，缓慢向 Actor 网络靠近。
* **Critic 网络（Q 网络）**：
	+ **作用**：估计当前状态 ss 和动作 aa 的 Q 值，即 Q(s,a)Q(s,a)，为 Actor 提供优化目标。
	+ **更新**：通过最小化与目标 Q 值的均方误差进行更新。
* **Target Critic 网络（目标 Q 网络）**：
	+ **作用**：生成 Q 值更新的目标，使得 Q 值更新更为稳定，减少振荡。
	+ **更新**：使用软更新，缓慢向 Critic 网络靠近。


大白话解释：


​ 1、DDPG实例化为actor，输入state输出action
​ 2、DDPG实例化为actor\_target
​ 3、DDPG实例化为critic\_target，输入next\_state和actor\_target(next\_state)经DQN计算输出target\_Q
​ 4、DDPG实例化为critic，输入state和action输出current\_Q，输入state和actor(state)【这个参数需要注意，不是action】经负均值计算输出actor\_loss


​ 5、current\_Q 和target\_Q进行critic的参数更新
​ 6、actor\_loss进行actor的参数更新


action实际上是batch\_action，state实际上是batch\_state，而**batch\_action !\= actor(batch\_state)**


因为actor是频繁更新的，而采样是随机采样，不是所有batch\_action都能随着actor的更新而同步更新


Critic网络的更新是一发而动全身的，相比于Actor网络的更新要复杂要重要许多




---


### 代码核心更新公式



* 上述代码与伪代码对应，意为计算预测Q值



* 上述代码与伪代码对应，意为使用均方误差损失函数更新Critic



![](https://img2024.cnblogs.com/blog/3382553/202410/3382553-20241018111326881-946848917.png)
* 上述代码与伪代码对应，意为使用确定性策略梯度更新Actor



* 上述代码与伪代码对应，意为使用策略梯度更新目标网络




---


**Actor和Critic的角色**：


* **Actor**：负责选择动作。它根据当前的状态输出一个确定性动作。
* **Critic**：评估Actor的动作。它通过计算状态\-动作值函数（Q值）来评估给定状态和动作的价值。


**更新逻辑**：


* **Critic的更新**：
	1. 使用经验回放缓冲区（Experience Replay）从中采样一批经验（状态、动作、奖励、下一个状态）。
	2. 计算目标Q值：使用目标网络（critic\_target）来估计下一个状态的Q值（target\_Q），并结合当前的奖励。
	3. 使用均方误差损失函数（MSELoss）来更新Critic的参数，使得预测的Q值（target\_Q）与当前Q值（current\_Q）尽量接近。
* **Actor的更新**：
	1. 根据当前的状态（state）从Critic得到Q值的梯度（即对Q值相对于动作的偏导数）。
	2. 使用确定性策略梯度（DPG）的方法来更新Actor的参数，目标是最大化Critic评估的Q值。




---


个人理解：


DQN算法是将q\_network中的参数每n轮一次复制到target\_network里面


DDPG使用系数τ来更新参数，将学习到的参数更加soft地拷贝给目标网络


DDPG采用了actor\-critic网络，所以比DQN多了两个网络


 \_\_EOF\_\_

   El Psy Kongroo!  - **本文链接：** [https://github.com/hassle/p/18473876](https://github.com)
 - **关于博主：** 研二计算机遥感方向转强化学习方向，喜欢英国源神、杀戮尖塔、香蕉锁头、galgame，和下午的一杯红茶。
 - **版权声明：** 本博客所有文章除特别声明外，均采用BY\-NC\-SA 许可协议。转载需要注明出处
 - **声援博主：** 点个赞再走吧，初音未来会护佑每一位虔诚的信徒！
     
