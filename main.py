import torch
import pickle
import math

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

def two_layer_net(inputs, target, params):
    #inputs : (1:N)
    #W1 : (N:D) b1 : (D)
    #W2 : (D:1) b2 : (1)
    Y1 = torch.mm(inputs, params['W1']) + params['b1']#(1,D)
    h1 = torch.clamp(Y1, min=0)#(1,D)
    Y2 = torch.mm(h1, params['W2']) + params['b2']#(1:1)
    out = torch.clamp(Y2, min=0)

    #for predict
    if target is None:
        return out

    loss = out - target

    grads = {}

    dout = 1

    #(1:D)
    mask = (out <= 0)
    dout[mask] = 0

    grads['W2'] = torch.mm(torch.t(h1), dout)#(D:1)
    grads['b2'] = torch.sum(dout, dim=0)
    dout = torch.mm(dout, torch.t(params['W2']))#(1:D)

    mask = (h1 <= 0)
    dout[mask] = 0

    grads['W1'] = torch.mm(torch.t(inputs), dout)#(N:D)
    grads['b1'] = torch.sum(dout, dim=0)
    dout = torch.mm(dout, torch.t(params['W1']))#(1:N)

    return loss, grads


def trainer(input_seq, target_seq, dimention, learning_rate, epoch):
    option_size = len(input_seq[0])
    params = {}
    params['W1'] = torch.randn(option_size, dimention) / (dimention**0.5)
    params['b1'] = torch.zeros(dimention)
    params['W2'] = torch.randn(dimention, 1)
    params['b2'] = torch.zeros(1)

    print("# of samples")
    print(len(input_seq))

    losses = []
    for _ in epoch:
        for i in range(len(input_seq)):
            loss, grad = two_layer_net(input_seq[i], target_seq[i], params)
            params['W1'] -= grad['W1']*learning_rate
            params['b1'] -= grad['b1']*learning_rate
            params['W2'] -= grad['W2']*learning_rate
            params['b2'] -= grad['b2']*learning_rate

            losses.append(loss.item())

            if (i%10000) == 0:
                avg_loss = sum(losses) / len(losses)
                print("%d  Loss : %f" %(i, avg_loss,))
                losses = []
    
    return params

def main():
    print('1. load data...')
    with open('datalist.data', 'rb') as f:
        data = pickle.load(f)

    print("total data : {}".format(len(data)))

    print('2. make input_seq, target_seq')
    input_seq = {}
    target_seq = {}
    for i in range(len(data)):
        input_seq[i] = torch.FloatTensor([data[i].get_satisNormalDist(), data[i].get_financialStatementNormalDist()])
        target_seq[i] = torch.FloatTensor(data[i].get_creditScore)

    print('3. training...')
    params = trainer(input_seq, target_seq, dimention=64, learning_rate=0.025, epoch=1)
    
    print('4. save data')
    with open('output.pickle', 'w') as f:
        f.write(params)
    
    print('5. finish...')

main()
