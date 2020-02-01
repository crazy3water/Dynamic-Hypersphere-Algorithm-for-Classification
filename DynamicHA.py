import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DHA():
    def __init__(self,DataFram,x_newW=None,test_per=0.1,R_constant = 1,lr=0.001,P=0.001,P_R2Cm=0.001,P_class=0.0001):
        """
        :param DataFram:  数据集
        :param test_per:  测试数据集的抽样比例
        :param R_constant:初始化半径
        :param x_newW:    W的变换维度
        :param lr:        优化器学习率
        :param P:         每个球体的惩罚系数
        :param P_R2Cm:    球体之间的惩罚系数
        :param P_class:   加速收敛系数
        """
        self.DataFram = DataFram
        self.x_newW = x_newW
        self.lr = lr
        self.P = P
        self.P_R2Cm = P_R2Cm
        self.P_class = P_class
        self.R_constant = R_constant
        print('-----------------------数据预处理----------------------')
        self.preprocess(test_per)

    #数据预处理
    def preprocess(self,per):
        unique =  self.DataFram.loc[:,0].unique()
        target = self.DataFram.loc[:,0].values
        data = self.DataFram.loc[:,1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).values

        if self.x_newW == None:
            n, self.x_w = data.shape
            self.x_newW = self.x_w
        else:
            n, self.x_w = data.shape
        random_index = np.random.choice(range(100), size=1, replace=False)
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=per,
                                                                          random_state=int(random_index))
        self.x_train_space = []
        for class_i in unique:
            self.x_train_space.append(x_train[(y_train==class_i)])
        print('------将训练数据集分为%d份'%(len(self.x_train_space)))

    def Acc_3(self, Cmpre_one, Cmpre_two, Cmpre_three, target):
        m, _ = np.shape(Cmpre_one)
        y_out = []
        y_ex = []
        with tf.name_scope('Acc'):
            y = np.zeros([m])
            for i in range(len(y)):
                if Cmpre_one[i] < 1 and Cmpre_two[i] > 1 and Cmpre_three[i] > 1:
                    y[i] = 1
                else:
                    if Cmpre_two[i] < 1 and Cmpre_one[i] > 1 and Cmpre_three[i] > 1:
                        y[i] = 2
                    else:
                        if Cmpre_three[i] < 1 and Cmpre_one[i] > 1 and Cmpre_two[i] > 1:
                            y[i] = 3
                        else:
                            if Cmpre_three[i] > 1 and Cmpre_one[i] > 1 and Cmpre_two[i] > 1:
                                y_out.append(i)
                            else:
                                y_ex.append(i)
                            if Cmpre_one[i] > Cmpre_two[i]:
                                if Cmpre_two[i] > Cmpre_three[i]:
                                    y[i] = 3
                                else:
                                    y[i] = 2
                            else:
                                if Cmpre_one[i] > Cmpre_three[i]:
                                    y[i] = 3
                                else:
                                    y[i] = 1
            correct_predict = np.equal(y, target)
            accuracy = np.mean(correct_predict)

        return accuracy

    #在圆心外惩罚
    def g1n_term(self, var, center, Rm, num):
        g1n = tf.linalg.norm(var - center) - Rm
        g1n_max = tf.where(tf.greater(g1n, 0), g1n, 0)
        penalty = tf.sum(g1n_max)  # if res>0,penalty = res else penalty = 0
        return penalty

    # 在圆心内惩罚
    def g2n_term(self, var, center, Rm, num):
        g2n = Rm - tf.linalg.norm(var - center)
        g2n_max = tf.where(tf.greater(g2n, 0), g2n, 0)
        penalty = tf.sum(g2n_max)   # if res>0,penalty = res else penalty = 0
        return penalty
    #产生模型
    def gen_model(self):
        circle = []
        r = []

        with tf.name_scope('Space'):
            with tf.name_scope('InitVariable'):
                #动态变量名设置
                with tf.name_scope('V'):
                    for i in range(len(self.x_train_space)):
                        exec('V{}={}'.format(i,tf.placeholder(tf.float32, shape=[None, self.x_w], name='Input{}'.format(i))))
                with tf.name_scope('TrainingStep'):
                    training_step = tf.Variable(5, name='TrainStep', trainable=False)
                    learning_rate = tf.Variable(self.lr, name='TrainStep', trainable=False)
                with tf.name_scope('weight'):
                    weight = tf.Variable(tf.random_normal([self.x_w, self.x_newW]), name='W', trainable=True)

                with tf.name_scope('bias'):
                    bias = tf.Variable(tf.zeros([self.x_newW]), name='B', trainable=True)

                with tf.name_scope('R'):
                    for i in range(len(self.x_train_space)):
                        exec('R{}={}'.format(i,tf.Variable(initial_value=self.R_constant, dtype=tf.float32, name='R'.format(i), trainable=True)))

            with tf.name_scope('layer'):
                for i in range(len(self.x_train_space)):
                    exec('U{}={}'.format(i, tf.matmul(exec('V{}'.format(i)), weight) + bias))

            for i in range(len(self.x_train_space)):
                with tf.name_scope('circle{}'.format(i)):
                    exec('Cm{}={}'.format(i, tf.mean(exec('V{}'.format(i)), axis=0)))


            with tf.name_scope('loss'):
                for i in range(len(self.x_train_space)):
                    exec('g1n{} = {}'.format(i,self.g1n_term(exec('U{}'.format(i)), exec('Cm{}'.format(i)), exec('R{}'.format(i)))))
                    exec('loss{}_pow =  {}'.format(i,tf.pow(exec('g1n{}'.format(i)), 2)
                                                   + tf.pow(exec('g2n{}'.format(i)), 2)
                                                   + tf.pow(exec('Rn{}'.format(i)), 2)))
                    g1n1 = self.g1n_term(U[:Nm1, :], Cm1, R1, Nm1)
                    g2n1 = self.g2n_term(U[Nm1:, :], Cm1, R1, Nm2 + Nm3)
                    Rn1 = tf.where(tf.greater(-R1, 0), -R1, 0)

                    # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }

                    loss1_pow = tf.pow(g1n1, 2) + tf.pow(g2n1, 2) + tf.pow(Rn1, 2)

                with tf.name_scope('loss_2'):
                    U_ = tf.concat([U[:Nm1, :], U[(Nm1 + Nm2):, :]], 0)
                    g1n2 = g1n_term(U[Nm1:Nm1 + Nm2, :], Cm2, R2, Nm2)
                    g2n2 = g2n_term(U_, Cm2, R2, Nm1 + Nm3)
                    Rn2 = tf.where(tf.greater(-R2, 0), -R2, 0)

                    # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }

                    loss2_pow = tf.pow(g1n2, 2) + tf.pow(g2n2, 2) + tf.pow(Rn2, 2)

                with tf.name_scope('loss_3'):
                    g1n3 = g1n_term(U[(Nm1 + Nm2):, :], Cm3, R3, Nm3)
                    g2n3 = g2n_term(U[:(Nm1 + Nm2), :], Cm3, R3, Nm1 + Nm2)
                    Rn3 = tf.where(tf.greater(-R3, 0), -R3, 0)

                    # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }
                    loss3_pow = tf.pow(g1n3, 2) + tf.pow(g2n3, 2) + tf.pow(Rn3, 2)

                with tf.name_scope('lossR2Cm_R12'):
                    Cm12_normal = (1 + 0) * (R1 + R2) - tf.linalg.norm(Cm1 - Cm2)
                    Cm12_penalty = tf.where(tf.greater(Cm12_normal, 0), Cm12_normal, 0)

                with tf.name_scope('lossR2Cm_R13'):
                    Cm13_normal = (1 + 0) * (R1 + R3) - tf.linalg.norm(Cm1 - Cm3)
                    Cm13_penalty = tf.where(tf.greater(Cm13_normal, 0), Cm13_normal, 0)

                with tf.name_scope('lossR2Cm_R23'):
                    Cm23_normal = (1 + 0) * (R2 + R3) - tf.linalg.norm(Cm2 - Cm3)
                    Cm23_penalty = tf.where(tf.greater(Cm23_normal, 0), Cm23_normal, 0)

            with tf.name_scope('loss_pow'):
                loss_pow = (loss1_pow + loss2_pow + loss3_pow)

            with tf.name_scope('loss_R2Cm'):
                loss_R2Cm = (Cm12_penalty + Cm13_penalty + Cm23_penalty)

                loss_class = tf.linalg.norm(U[:Nm1, :] - Cm1) + tf.linalg.norm(U[Nm1:Nm1 + Nm2, :] - Cm2) + \
                             tf.linalg.norm(U[(Nm1 + Nm2):, :] - Cm2)

            with tf.name_scope('loss_all'):
                loss_all = P * (loss_pow) + P_R2Cm * loss_R2Cm + P_class * loss_class

            with tf.name_scope('train'):
                learning_rate = tf.train.exponential_decay(learning_rate, training_step, decay_steps=100,
                                                           decay_rate=0.8)
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)

                loss_list = []
                R1_list = []
                R2_list = []
                R3_list = []
                print('Enter train the Space........')

                for j in range(step):
                    _ = sess.run(train_op, feed_dict={V: datatrain})
                    loss = sess.run(loss_all, feed_dict={V: datatrain})
                    loss_list.append(loss)
                    R_1, R_2, R_3 = sess.run((R1, R2, R3))
                    R1_list.append(R_1)
                    R2_list.append(R_2)
                    R3_list.append(R_3)
                print(loss_list)
                with open('loss_wine_.txt', 'w') as f:
                    for l in loss_list:
                        f.write(str(l) + ',')
                f.close()
                weight_ = sess.run(weight)
                bias_ = sess.run(bias)
                Circle1 = sess.run(Cm1, feed_dict={V: datatrain})
                R_1 = sess.run(R1)
                Cmpre_one = Cm_test_distance(datatest, Circle1, weight_, bias_, R_1)
                gaus_Cmpre_one = Gaussian_PDF(weight_, bias_, datatrain[:Nm1, :], datatest)
                circle.append(Circle1)
                r.append(R_1)

                Circle2 = sess.run(Cm2, feed_dict={V: datatrain})
                R_2 = sess.run(R2)
                Cmpre_two = Cm_test_distance(datatest, Circle2, weight_, bias_, R_2)
                gaus_Cmpre_two = Gaussian_PDF(weight_, bias_, datatrain[Nm1:(Nm1 + Nm2), :], datatest)
                circle.append(Circle2)
                r.append(R_2)

                Circle3 = sess.run(Cm3, feed_dict={V: datatrain})
                R_3 = sess.run(R3)
                Cmpre_three = Cm_test_distance(datatest, Circle3, weight_, bias_, R_3)
                gaus_Cmpre_three = Gaussian_PDF(weight_, bias_, datatrain[(Nm1 + Nm2):, :], datatest)
                circle.append(Circle3)
                r.append(R_3)

                Acc = Acc_3(Cmpre_one, Cmpre_two, Cmpre_three, target_test)
                gaus_acc = Gaussian_acc3(gaus_Cmpre_one, gaus_Cmpre_two, gaus_Cmpre_three, target_test)
                if Acc < gaus_acc:
                    Acc = gaus_acc
                print(random_index)
                print(Acc)

            print('end')
        return Acc

if __name__ == "__main__":
    DataPath = r'.\Wine.csv'
    DataSet = pd.read_csv(DataPath)



