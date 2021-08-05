import numpy as np
import matplotlib.pyplot as plt



def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#plt.figure(figsize=(15, 10))

#DQN_target
# scores_dqn_target = np.loadtxt("E:/Dual Memory SAR/DQN_DEMO_EFAZ/scores_.txt")

# #C:\Users\Towsif\Desktop\New folder\DQNS\dqn_target
# scores_ma_dqn_target = moving_average(scores_dqn_target, n = 100)

# scores_ma_dqn_target_1000 = moving_average(scores_dqn_target, n = 1000)

# plt.plot(np.arange(len(scores_ma_dqn_target)), scores_ma_dqn_target, alpha = 0.1, color = 'b')
# plt.plot(np.arange(len(scores_ma_dqn_target_1000)), scores_ma_dqn_target_1000, alpha = 1, color = 'b', label = "DQN_target_mem")

# #DQN_target_mem
scores_dqn_target_mem = np.loadtxt("E:/GymAI/Lunarlander/DQN_target/scores_dqn_classic.txt")
scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'g',label ="DQN Youme")


scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM - Copy/scores_95_randomized.txt")
scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'b',label =" NEW DQN")


# plt.plot(np.argmax(scores_ma_dqn_target_mem), 'ro')

# plt.axhline(y=max(scores_ma_dqn_target_mem), color='y', linestyle='-')
# plt.axhline(y=200, color='g', linestyle='-')


# scores_dqn_target_mem = np.loadtxt("E:/GymAI/Lunarlander/DQN_target/scores_50.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'g',label ="DQN 2")
#plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem)), scores_ma_dqn_target_1000_mem, color = 'g', label = "DDQN")


# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_50.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'r',label = "DQN Score 50")

# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_65.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'b',label = "DQN Score 65")

# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_70.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'orange',label = "DQN Score 70 Towsif")

# scores_dqn_target_mem_new = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_random_alpha_0.05.txt")
# scores_ma_dqn_target_mem_new = moving_average(scores_dqn_target_mem_new, n = 100)
# scores_ma_dqn_target_1000_mem_new = moving_average(scores_dqn_target_mem_new, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem_new)), scores_ma_dqn_target_mem_new, color = 'purple',label="DQN Reward 70 youme")
# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores_85.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'blue',label = "DQN Score 85")

# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores_90.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'orange',label = "DQN Score 90")

# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_dqn.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem_e = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'r',label = "DQN Score Towsif")

# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_100_Towsif.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'blue',label = "DQN Score towsif")

# scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores_100_A.txt")
# scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'cyan',label = "DQN Score Youme")


# # scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores_100_again.txt")
# # scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# # scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# # plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'cyan',label = "DQN Score 100")

# # scores_dqn_target_mem = np.loadtxt("E:/GYM DUAL MEMORY/demo_DQN_DUAL_MEM/scores/scores_75_alvi.txt")
# # scores_ma_dqn_target_mem = moving_average(scores_dqn_target_mem, n = 100)
# # scores_ma_dqn_target_1000_mem = moving_average(scores_dqn_target_mem, n = 1000)
# # plt.plot(np.arange(len(scores_ma_dqn_target_mem)), scores_ma_dqn_target_mem, color = 'pink',label = "DQN Score 100")

# #plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem_new)), scores_ma_dqn_target_1000_mem_new, color = 'b', label = "DDQN with Dual Memory")

# #DQN_target_mem_randomized
# # scores_dqn_target_mem_randomized = np.loadtxt("C:/Users/Towsif/Desktop/499 code/Dual Memory DQNS/DDQN dual mem/DQN_DUAL_MEM_Randomized/scores_dqn.txt")
# # maxs = np.max(scores_dqn_target_mem_randomized)
# # print('rand',maxs)
# # scores_ma_dqn_target_mem_randomized = moving_average(scores_dqn_target_mem_randomized, n = 100)
# # maxs = np.max(scores_ma_dqn_target_mem_randomized)
# # print('rand moving avg 100',maxs)
# # scores_ma_dqn_target_1000_mem_randomized = moving_average(scores_dqn_target_mem_randomized, n = 1000)
# # maxs = np.max(scores_ma_dqn_target_1000_mem_randomized)
# # print('rand moving avg 1000',maxs)
# # plt.plot(np.arange(len(scores_ma_dqn_target_mem_randomized)), scores_ma_dqn_target_mem_randomized, alpha = 0.1, color = 'r')
# # plt.plot(np.arange(len(scores_ma_dqn_target_1000_mem_randomized)), scores_ma_dqn_target_1000_mem_randomized, alpha = 1, color = 'r', label = "DQN_target_mem_randomized")


plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.legend()
plt.savefig('DQN_graph.png')
plt.show()






# scores_dqn = np.loadtxt("scores_100.txt")
# scores_ma_dqn = moving_average(scores_dqn, n=10)
# scores_ma_dqn_1000 = moving_average(scores_dqn, n=100)

# plt.plot(np.arange(len(scores_ma_dqn)), scores_ma_dqn,
#          alpha=0.2, color='b')
# plt.plot(np.arange(len(scores_ma_dqn_1000)),
#          scores_ma_dqn_1000, label="DQN", color='b')
