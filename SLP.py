import torch
import pickle
import random

class DataClass:
    def __init__(self):
        self.name=None
        self.yearMon=None
        self.rate=None
        self.satis=None
        self.satisNormalDist=None
        self.sizeOfCompany=None
        self.bonji_g=None
        self.budoRate=None
        self.classification=None
        self.isBig=None
        self.sangJang=None
        self.stateCode=None
        self.financialStatement=None
        self.financialStatementNormalDist=None
        self.__nonFinancialStatement=None
        self.classFinacial=None
        self.__classNonfinancialStatement=None
        self.creditScore=None
        self.classCredit=None

    def printData(self):
        print("name : {}\nyearMon : {}\nrate : {}\nsatis : {}\nsatisNormalDist : {}\nsizeOfCompany : {}\nbonji_g : {}\nbudoRate : {}\nclassification : {}\nisBig : {}\nsangJang : {}\nstateCode : {}\nfinancialStatement : {}\nfinancialStatementNormalDist : {}\n__nonFinancialStatement : {}\nclassFinancial : {}\n__classNonfinancialStatement : {}\ncreditScore : {}\nclassCredit : {}\n".format(self.name, self.yearMon, self.rate, self.satis, self.satisNormalDist, self.sizeOfCompany, self.bonji_g, self.budoRate, self.classification, self.isBig, self.sangJang, self.stateCode, self.financialStatement, self.financialStatementNormalDist, self.__nonFinancialStatement, self.classFinacial, self.__classNonfinancialStatement, self.creditScore, self.classCredit))

    def get_name(self):
        return self.name

    def get_yearMon(self):
        return self.yearMon

    def get_rate(self):
        return self.rate

    def get_satis(self):
        return self.satis

    def get_satisNormalDist(self):
        return self.satisNormalDist

    def get_sizeOfCompany(self):
        return self.sizeOfCompany

    def get_bonji_g(self):
        return self.bonji_g

    def get_budoRate(self):
        return self.budoRate

    def get_classification(self):
        return self.classification

    def get_isBig(self):    
        self.isBig
    
    def get_sangJang(self):
        return self.sangJang

    def get_stateCode(self):
        return self.stateCode

    def get_financialStatement(self):
        return self.financialStatement

    def get_financialStatementNormalDist(self):
        return self.financialStatementNormalDist

    def get_nonFiancialStatement(self):
        return self.__nonFinancialStatement

    def get_classFinacial(self):
        return self.classFinacial

    def get_classNonfinancialStatement(self):
        return self.__classNonfinancialStatement

    def get_creditScore(self):
        return self.creditScore

    def get_classCredit(self):
        return self.classCredit

def wall():
    print("================================================")

def one_layer_net(inputs, target, active, params):
    #inputs : (1, n)
    #W1 : (N, 1)
    #b1 : (1, 1)
    Y = torch.mm(inputs, params['W1'][active]) + params['b1']
    #Y : (1, 1)
    if target is None:
        return Y

    loss = (Y - target)**2 / 2

    grads = {}
    grads['W1'] = (Y - target)*inputs
    grads['b1'] = Y - target

    return loss, grads

def SLP_trainer(input_seq, target_seq, active_seq, learning_rate, decay, epoch):
    option_size = 7
    params = {}
    params['W1'] = torch.randn(option_size, 1)
    params['b1'] = torch.zeros(1, 1)

    losses = []
    decay_rate = 1
    for e in range(epoch):
        for i in range(len(input_seq)):
            loss, grad = one_layer_net(input_seq[i], target_seq[i], active_seq[i], params)
            params['W1'][active_seq[i]] -= torch.t(grad['W1'])*learning_rate*decay_rate
            params['b1'] -= grad['b1']*learning_rate*decay_rate

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print("epoch : %d, Loss : %f" %(e, avg_loss,))
        losses = []
        decay_rate = decay_rate*decay
    return params

def main():
    wall()
    print('1. load data...')
    with open('datalist.data', 'rb') as f:
        data = pickle.load(f)

    print("total data : {}".format(len(data)))

    random.shuffle(data)

    wall()
    print('2. make input_seq, test_seq')
    input_seq = {}
    target_seq = {}
    active_seq = {}
    check_list = {}

    test_seq = {}
    test_target_seq = {}
    test_active_seq = {}
    test_check_list = {}
    
    i = 0
    j = 0
    test_set_num = 10000
    for d in range(len(data)):
        input_box = []
        active = []
        check = ''
        if data[d].get_satisNormalDist()!=None:
            input_box.append(data[d].get_satisNormalDist())
            active.append(0)
            check += '0'
        
        if data[d].get_rate()!=None:
            input_box.append(data[d].get_rate())
            active.append(1)
            check += '1'

        if data[d].get_budoRate()!=None:
            input_box.append(data[d].get_budoRate())
            active.append(2)
            check += '2'

        if data[d].get_isBig()!=None:
            input_box.append(data[d].get_isBig())
            active.append(3)
            check += '3'
        
        if data[d].get_financialStatementNormalDist()!=None:
            input_box.append(data[d].get_financialStatementNormalDist())
            active.append(4)
            check += '4'

        if data[d].get_classification()!=None:
            input_box.append(data[d].get_classification()//10000)
            input_box.append((data[d].get_classification()//1000)%10)
            active.append(5)
            active.append(6)
            check += '56'

        if j<test_set_num:
            test_seq[j] = torch.FloatTensor(input_box).unsqueeze(0)
            test_target_seq[j] = torch.FloatTensor([data[d].get_creditScore()])
            test_active_seq[j] = active
            test_check_list[j] = check
            j += 1
        else:
            input_seq[i] = torch.FloatTensor(input_box).unsqueeze(0)
            target_seq[i] = torch.FloatTensor([data[d].get_creditScore()])
            active_seq[i] = active
            check_list[i] = check
            i += 1

    print("# of input sample : {}".format(len(input_seq)))
    print("# of test sample : {}".format(len(test_seq)))

    wall()
    print('3. training...')
    lr = 0.0001
    dc = 0.98
    ep = 10
    print('learning_rate : {}'.format(lr))
    print('decay : {}'.format(dc))
    print('epoch : {}'.format(ep))
    params = SLP_trainer(input_seq, target_seq, active_seq, learning_rate=lr, decay=dc, epoch=ep)
    
    wall()
    print('4. save data...')
    with open('params.pickle', 'wb') as f:
        pickle.dump(params, f)

    input_data = {'input_seq':input_seq, 'target_seq':target_seq, 'active_seq':active_seq, 'check_list':check_list}
    with open('input.pickle', 'wb') as f:
        pickle.dump(input_data, f)
    
    test = {'test_seq':test_seq, 'test_target_seq':test_target_seq, 'test_active_seq':test_active_seq, 'test_check_list':test_check_list}
    with open('test.pickle', 'wb') as f:
        pickle.dump(test, f)
    
    error = []
    wall()
    print('6. test...')
    for i in range(len(test_seq)):
        error.append(((test_target_seq[i] - one_layer_net(test_seq[i], None, test_active_seq[i], params))**2)**0.5)
    print("average error : {}".format(sum(error, 0.0)/len(error)))
    wall()
    print("7. finish")
    wall()

def test():
    with open('output.pickle', 'rb') as f:
        params = pickle.load(f)

    with open('test.pickle', 'rb') as f:
        test = pickle.load(f)

    error = []
    for i in range(len(test['test_seq'])):
        error.append(((test['test_target_seq'][i] - one_layer_net(test['test_seq'][i], None, test['test_active_seq'][i], params))**2)**0.5)
    print("average error : {}".format(sum(error, 0.0)/len(error)))


main()