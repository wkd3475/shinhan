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

"""
def two_layer_net(inputs, target, active, params):
    #inputs : (1:N)
    #W1 : (N:D) b1 : (D)
    #W2 : (D:1) b2 : (1)
    #active : n

    Y1 = torch.mm(inputs, params['W1'][active]) + params['b1']#(1,D)
    h1 = torch.clamp(Y1, min=0)#(1,D)
    Y2 = torch.mm(h1, params['W2']) + params['b2']#(1:1)

    #for predict
    if target is None:
        return Y2

    loss = (Y2 - target)**2

    grads = {}

    #(1:D)

    grads['W2'] = (Y2-target)*inputs
    #grads['b2'] = ??
    dout = torch.mm(dout, torch.t(params['W2']))#(1:D)

    mask = (h1 <= 0)
    dout[mask] = 0

    grads['W1'] = torch.mm(torch.t(inputs), dout)#(N:D)
    grads['b1'] = torch.sum(dout, dim=0)
    dout = torch.mm(dout, torch.t(params['W1']))#(1:N)

    return loss, grads
"""

def trainer(input_seq, target_seq, active_seq, dimention, learning_rate, decay, epoch):
    option_size = 7
    params = {}
    #params['W1'] = torch.randn(option_size, dimention) / (dimention**0.5)
    #params['b1'] = torch.zeros(dimention)
    params['W1'] = torch.randn(option_size, 1)
    params['b1'] = torch.zeros(1, 1)
    #params['W2'] = torch.randn(dimention, 1)
    #params['b2'] = torch.zeros(1)

    print("# of samples")
    print(len(input_seq))

    losses = []
    for e in range(epoch):
        for i in range(len(input_seq)):
            loss, grad = one_layer_net(input_seq[i], target_seq[i], active_seq[i], params)
            params['W1'][active_seq[i]] -= torch.t(grad['W1'])*learning_rate*decay
            params['b1'] -= grad['b1']*learning_rate*decay
            #params['W2'] -= grad['W2']*learning_rate
            #params['b2'] -= grad['b2']*learning_rate

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print("epoch : %d, Loss : %f" %(e, avg_loss,))
        losses = []
        decay = decay**2
    return params

def main():
    print('1. load data...')
    with open('datalist.data', 'rb') as f:
        data = pickle.load(f)

    print("total data : {}".format(len(data)))

    print('2. make input_seq, target_seq')
    input_seq = {}
    target_seq = {}
    active_seq = {}
    for i in range(len(data)):
        input_box = []
        active = []

        if data[i].get_satisNormalDist()!=None:
            input_box.append(data[i].get_satisNormalDist())
            active.append(0)
        
        if data[i].get_rate()!=None:
            input_box.append(data[i].get_rate())
            active.append(1)

        if data[i].get_budoRate()!=None:
            input_box.append(data[i].get_budoRate())
            active.append(2)

        if data[i].get_isBig()!=None:
            input_box.append(data[i].get_isBig())
            active.append(3)
        
        if data[i].get_financialStatementNormalDist()!=None:
            input_box.append(data[i].get_financialStatementNormalDist())
            active.append(4)

        if data[i].get_classification()!=None:
            input_box.append(data[i].get_classification()//10000)
            active.append(5)
            input_box.append((data[i].get_classification()//1000)%10)
            active.append(6)

        input_seq[i] = torch.FloatTensor(input_box).unsqueeze(0)
        target_seq[i] = torch.FloatTensor([data[i].get_creditScore()])
        active_seq[i] = active

    print('3. training...')
    params = trainer(input_seq, target_seq, active_seq, dimention=64, learning_rate=0.0001, decay=0.98, epoch=10)
    
    print('4. save data...')
    with open('output.pickle', 'wb') as f:
        pickle.dump(params, f)
    
    print('5. finish...')

def test():
    with open('output.pickle', 'rb') as f:
        params = pickle.load(f)

    with open('datalist.data', 'rb') as f:
        data = pickle.load(f)

    random.shuffle(data)

    input_seq = {}
    target_seq = {}
    active_seq = {}
    for i in range(10):
        input_box = []
        active = []

        if data[i].get_satisNormalDist()!=None:
            input_box.append(data[i].get_satisNormalDist())
            active.append(0)
        
        if data[i].get_rate()!=None:
            input_box.append(data[i].get_rate())
            active.append(1)

        if data[i].get_budoRate()!=None:
            input_box.append(data[i].get_budoRate())
            active.append(2)

        if data[i].get_isBig()!=None:
            input_box.append(data[i].get_isBig())
            active.append(3)
        
        if data[i].get_financialStatementNormalDist()!=None:
            input_box.append(data[i].get_financialStatementNormalDist())
            active.append(4)

        if data[i].get_classification()!=None:
            input_box.append(data[i].get_classification()//10000)
            active.append(5)
            input_box.append((data[i].get_classification()//1000)%10)
            active.append(6)

        input_seq[i] = torch.FloatTensor(input_box).unsqueeze(0)
        target_seq[i] = torch.FloatTensor([data[i].get_creditScore()])
        active_seq[i] = active

        print("real : %f, predict : %f" %(target_seq[i], one_layer_net(input_seq[i], None, active_seq[i], params)))

test()
