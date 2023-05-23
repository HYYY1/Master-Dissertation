import numpy as np
np.random.seed(1234)
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras import optimizers, losses, metrics, models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
import random

# 不显示警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# stock_list = {"Apple": 3400,"Amazon": 3400,"Microsoft": 3400, "Google": 2800, "Tesla": 1800}
stock_list = {"Microsoft": 3400}

best_model_result = 0
best_gene = []  # 初始化一个保存最优参数组合的列表
stock_name = ""
s_length = 0


def load_data():
    target_stock = pd.read_csv("./data/n_" + stock_name + ".csv")
    target_stock = pd.DataFrame(target_stock)
    # 时间点长度
    time_stamp = 50
    # 划分训练集与验证集
    target_stock = target_stock[['Open', 'High', 'Low', 'Close', 'Volume']]  # 'Volume'

    # 新增一列正负表示涨跌
    close = target_stock['Close'].tolist()
    y = []
    for i in range(len(target_stock) - 1):
        if close[i + 1] >= close[i]:
            y.append(1)
        else:
            y.append(-1)
    y.append(0)

    v4 = []
    for i in range(len(target_stock) - time_stamp - 1):
        if (y[i + time_stamp - 2] == 1):
            v4.append(1)
        else:
            v4.append(0)
    v4.append(0)
    target_stock["trend"] = y
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_stock)

    train = scaled_data[0:s_length + time_stamp]
    test = scaled_data[s_length - time_stamp:]
    # 训练集
    x_train, y_train = [], []
    for i in range(len(train) - time_stamp):
        train[i + time_stamp - 1][5] = v4[i]
        x_train.append(train[i:i + time_stamp])
        y_train.append(train[i + time_stamp, 3])

    x_train, y_train = np.array(x_train), np.array(y_train)
    # 测试集
    x_test, y_test = [], []
    for i in range(len(test) - time_stamp):
        test[i + time_stamp - 1][5] = v4[i + s_length - time_stamp]
        x_test.append(test[i:i + time_stamp])
        y_test.append(test[i + time_stamp, 3])

    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, x_test, y_train, y_test


def transncomp(closing_price, y_valid):
    # preprocessing,ues list
    y_valid = y_valid.reshape(-1)
    closing_price = np.array(closing_price)
    closing_price.reshape(-1)

    # temp1
    y_valid.tolist()
    temp1 = []
    for i in range(len(y_valid) - 1):
        if y_valid[i + 1] >= y_valid[i]:
            temp1.append(1)
        else:
            temp1.append(-1)

    # temp2
    closing_price.tolist()
    temp2 = []
    for i in range(len(closing_price) - 1):
        if closing_price[i + 1] >= closing_price[i]:
            temp2.append(1)
        else:
            temp2.append(-1)

    # compare
    sum = 0
    for i, j in zip(temp1, temp2):
        if i == j:
            sum += 1
    acc = sum / len(y_valid)
    return acc


def lstm_mode(inputs, units_num, sequences_state):
    # input主要是用来定义lstm的输入，input的一般是在第一层lstm层之前，units_num即是隐藏层神经元个数，sequence_state即是lstm层输出的方式
    lstm = LSTM(units_num, return_sequences=sequences_state)(inputs)
    # print("lstm:", lstm.shape)
    return lstm


# 定义全连接层、BN层
def dense_mode(input, dropout, units_num):
    # 这里主要定义全连接层的输入，input参数定义dense的第一次输入，units_num代表隐藏层神经元个数
    # 这里定义全连接层，采用L2正则化来防止过拟合，激活函数为relu
    dense = Dense(units_num, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='tanh')(input)
    # print("dense：", dense.shape)
    # 定义dropout层，概率为0.2
    drop_out = Dropout(rate=dropout)(dense)
    # 定义BN层，可以理解为是隐藏层的标准化过程
    # dense_bn = BatchNormalization()(drop_out)
    return dense, drop_out


# 这里定义的即是评价lstm效果的函数——也是遗传算法的适应度函数
def aim_function(x_train, y_train, x_test, y_test, num):
    # 这里传入数据和参数数组num,num保存了需要优化的参数
    # 这里我们设置num数组中num[0]代表lstm的层数。
    global best_model_result
    lstm_layers = num[0]
    # num[2:2 + lstm_layers]分别为lstm各层的神经元个数，num(1)为全连接层的层数)
    lstm_units = num[2:2 + lstm_layers]
    # 将num
    lstm_name = list(np.zeros((lstm_layers,)))
    # 设置全连接层的参数
    # num(1)为全连接的参数
    lstm_dense_layers = num[1]
    # 将lstm层之后的地方作为全连接层各层的参数
    lstm_dense_units = num[2 + lstm_layers: 2 + lstm_layers + lstm_dense_layers]
    #
    lstm_dense_name = list(np.zeros((lstm_dense_layers,)))
    lstm_dense_dropout_name = list(np.zeros((lstm_dense_layers,)))

    dropout = num[2 + lstm_layers + lstm_dense_layers]
    epochs = int(num[2 + lstm_layers + lstm_dense_layers + 1])

    # 这主要是定义lstm的第一层输入，形状为训练集数据的形状
    inputs_lstm = Input(shape=(x_train.shape[1], x_train.shape[2]))

    # 这里定义lstm层的输入（如果为第一层lstm层，则将初始化的input输入，如果不是第一层，则接受上一层输出的结果）
    for i in range(lstm_layers):
        if i == 0:
            inputs = inputs_lstm
        else:
            inputs = lstm_name[i - 1]
        if i == lstm_layers - 1:
            sequences_state = False
        else:
            sequences_state = True
        # 通过循环，我们将每层lstm的参数都设计完成
        lstm_name[i] = lstm_mode(inputs, lstm_units[i], sequences_state=sequences_state)

    # 同理设计全连接层神经网络的参数
    for i in range(lstm_dense_layers):
        if i == 0:
            inputs = lstm_name[lstm_layers - 1]
        else:
            inputs = lstm_dense_name[i - 1]
        lstm_dense_name[i], lstm_dense_dropout_name[i] = dense_mode(inputs, dropout, units_num=lstm_dense_units[i])

    outputs_lstm = Dense(1)(lstm_dense_dropout_name[lstm_dense_layers - 1])
    # print("last_dense", outputs_lstm.shape)
    # 利用函数式调试神经网络，调用inputs和outputs之间的神经网络
    LSTM_model = tf.keras.Model(inputs=inputs_lstm, outputs=outputs_lstm)
    LSTM_model.compile(optimizer=optimizers.Adam(), loss='mean_squared_error', )
    # print("训练集形状", x_train.shape)

    history = LSTM_model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.2, verbose=1)
    # 验证模型,model.evaluate返回的值是一个数组，其中score[0]为loss,score[1]为准确度

    # 反归一化
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit_transform(pd.DataFrame(test['Close'].values))
    closing_price = LSTM_model.predict(x_test)

    # closing_price = scaler.inverse_transform(closing_price)
    # y_valid = scaler.inverse_transform([y_test])
    # acc
    acc = transncomp(closing_price, y_test)

    # 保存最优模型，并用最优模型的参数设置代替下一代最弱参数
    if acc > best_model_result:
        best_model_result = acc
        LSTM_model.save("AGA_" + stock_name + '.h5')

    # 原始的并没有保存模型
    return acc


# 设置遗传算法的参数
DNA_size = 2
DNA_size_max = 8  # 每条染色体的长度
POP_size = 50  # 种群数量
CROSS_RATE = 0.5  # 交叉率
MUTATION_RATE = 0.01  # 变异率
N_GENERATIONS = 50  # 迭代次数

Pc1 = 0.9  # 自适应交叉概率1
Pc2 = 0.6  # 自适应交叉概率2
Pm1 = 0.1  # 自适应变异概率1
Pm2 = 0.5  # 自适应变异概率2


# 定义适用度函数，即aim_function函数，接收返回值
def get_fitness(x):
    return aim_function(x_train, y_train, x_test, y_test, num=x)


# 生成新的种群
def select(pop, fitness):
    # 这里主要是进行选择操作，即从20个种群中随机选取重复随机采样出20个种群进行种群初始化操作，p代表被选择的概率，这里采用的是轮盘赌的方式
    idx = np.random.choice(np.arange(POP_size), size=POP_size, replace=True, p=fitness / fitness.sum())
    # 将选择的种群组成初始种群pop
    return pop[idx]


# 交叉函数
def crossover(parent, pop, cross_rate):
    # 这里主要进行交叉操作，随机数小于交叉概率则发生交叉
    if np.random.rand() < cross_rate:
        # 从20个种群中选择一个种群进行交叉
        i_ = np.random.randint(0, POP_size, size=1)  # 染色体的序号
        # 这里将生成一个8维的2进制数，并转换层成bool类型，true表示该位置交叉，False表示不交叉
        cross_points = np.random.randint(0, 2, size=DNA_size_max).astype(np.bool)  # 用True、False表示是否置换

        # 这一部分主要是对针对不做变异的部分
        for i, point in enumerate(cross_points):
            '''
            第一部分这里是指该位点为神经元个数的位点，本来该交换，但其中位点为0,末尾的0位置就
            不应该交叉，因为交叉完后,会对前两位的参数产生影响。

            第二部分即对前两位不进行交叉操作，因为前两位代表的是层数，层数交叉后会对神经元的个数产生影响
            '''
            # 第一部分
            if point == True and pop[i_, i] * parent[i] == 0:
                cross_points[i] = False
            # 第二部分
            if point == True and i < 2:
                cross_points[i] = False
        # 将第i_条染色体上对应位置的基因置换到parent染色体上
        parent[cross_points] = pop[i_, cross_points]
    return parent


# 定义变异函数
def mutate(child, mutate_rate):
    # mutate - k2 = 0.5 k4 = 0.5

    # 变异操作也只是针对后6位参数
    for point in range(DNA_size_max):
        if np.random.rand() < mutate_rate * 0.5:
            # 2位参数之后的参数才参与变异
            if (point >= 2) and (point < len(child) - 2):
                if child[point] != 0:
                    child[point] = np.random.randint(32, 128)
            if point == len(child) - 2:
                child[point] = np.random.randint(1, 5) / 10
            if point == len(child) - 1:
                child[point] = np.random.randint(1, 3)
    return child


def go():
    global best_model_result
    global best_gene
    best_model_result = 0
    best_gene = []  # 初始化一个保存最优参数组合的列表
    # 在数组中保存每轮最优，输出数据
    best = []
    best.append(0)
    pop_pop = []  # 记录一个列表保存参数和适应度 画热力图
    # 初始化2列层数参数
    pop_layers = np.zeros((POP_size, DNA_size), np.int32)
    pop_layers[:, 0] = np.random.randint(1, 3, size=(POP_size,))  # change 4
    pop_layers[:, 1] = np.random.randint(1, 3, size=(POP_size,))

    # 种群
    # 初始化20x8的种群
    pop = np.zeros((POP_size, DNA_size_max))
    # 将初始化的种群赋值，前两列为层数参数，后6列为神经元个数参数
    for i in range(POP_size):
        # 随机从[32,256]中抽取随机数组组成神经元个数信息
        pop_neurons = np.random.randint(32, 128, size=(pop_layers[i].sum(),))
        # 将2列层数信息和6列神经元个数信息合并乘8维种群信息
        pop_stack = np.hstack((pop_layers[i], pop_neurons))
        # 将这些信息赋值给pop种群进行初始化种群
        for j, gene in enumerate(pop_stack):
            pop[i][j] = gene

    # 在迭代次数内，计算种群的适应度函数
    for each_generation in range(N_GENERATIONS):
        # 初始化适应度
        fitness = np.zeros([POP_size, ])
        # 遍历20个种群，对基因进行操作
        for i in range(POP_size):
            # 第i个染色体上的基因
            pop_list = list(pop[i])
            pop_copy = np.array(pop_list).copy()
            # 对赋值为0的基因进行删除
            for j, each in enumerate(pop_list):
                if each == 0.0:
                    index = j
                    pop_list = pop_list[:j]
            # 将基因进行转换为int类型
            for k, each in enumerate(pop_list):
                each_int = int(each)
                pop_list[k] = each_int
            # 将计算出来的适应度填写在适应度数组中

            # test1 add dropout and epochs
            pop_list.append(random.randint(1, 5) / 10)
            pop_list.append(random.randint(1, 3))

            fitness[i] = get_fitness(pop_list)
            # 输出结果
            print('第%d代第%d个染色体的适应度为%f' % (each_generation + 1, i + 1, fitness[i]))
            print('此染色体为：', pop_list)
            pop_copy = pop_copy.tolist()
            pop_copy.append(fitness[i])
            pop_pop.append(pop_copy)

        # 创建最优模型
        print('Generation:', each_generation + 1, 'Most fitted DNA:', pop[np.argmax(fitness), :], '适应度为：',
              fitness[np.argmax(fitness)])
        # 记录每代最好值，如果当代不如上一代，则替换
        best.append(best_model_result)
        if best_model_result > best[-2] or len(best_gene) == 1:
            best_gene = []
            for i in range(DNA_size_max):
                best_gene.append(pop[np.argmax(fitness), :][i])
        else:
            if best_model_result < best[-2] and each_generation > 20:
                pop[np.argmin(fitness), :] = [i for i in best_gene]
                fitness[np.argmin(fitness)] = best_model_result

        # 计算自适应概率,这里可以采用序优化的方法，形成sns-aga
        # 产生一个带有适应度值和平均值的新列表
        fitness_new = list(fitness)
        fitness_new.append(np.mean(fitness))
        fitness_new.sort()
        # 获取序号
        for j in range(POP_size + 1):
            if fitness_new[j] == np.mean(fitness):
                N2 = j + 1
        N3 = len(fitness_new)
        index_array = []
        cross_rate = []
        mutate_rate = []

        for i in fitness_new:
            if i >= np.mean(fitness):
                index_array.append(0)
            else:
                index_array.append(1)

        # cross - Pc1 Pc2
        for i in range(POP_size + 1):
            if index_array[i] == 0:
                cross_rate.append(np.reciprocal(((Pc1 - Pc2) + np.exp((i + 1 - N2) / (N3 - N2))) * Pc1))
            else:
                cross_rate.append(0.9)
        del (cross_rate[N2])
        # mutate - Pm1 Pm2
        for i in range(POP_size):
            if index_array[i] == 0:
                mutate_rate.append(np.reciprocal(((Pm1 - Pm2) + np.exp((i + 1 - N2) / (N3 - N2))) * Pm1))
            else:
                mutate_rate.append(0.09)

        # 生成新的种群
        pop = select(pop, fitness)
        # 复制一遍种群
        pop_copy = pop.copy()
        # 遍历pop中的每一个种群，进行交叉，变异，遗传操作
        index = 0
        for parent in pop:
            child = crossover(parent, pop_copy, cross_rate[index])
            child = mutate(child, mutate_rate[index])
            parent = child
            index += 1

    print(best)
    # 需要保存的数据
    csv1 = pd.DataFrame(best)
    csv1 = csv1.to_csv("AGA_" + stock_name + "_result.csv")

    csv2 = pd.DataFrame(pop_pop)
    csv2 = csv2.to_csv("AGA_" + stock_name + "_pop.csv")


for key, value in stock_list.items():
    print(key, "------------", value)
    stock_name = key
    s_length = value
    x_train, x_test, y_train, y_test = load_data()
    go()
