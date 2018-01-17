import numpy as np
import pandas as pd
import itertools

CFG = []
with open("grammar.txt") as f:
    for line in f:
        CFG.append(line.strip("\n").split("->"))
        
Dict = pd.DataFrame(CFG)


class CYK():
    
    def __init__(self,Dict):
        self.Dict = Dict
    def isINCFL(self,x):
            n = len(x)
            self.tableau = np.empty((n,n),dtype=object)
            for i in range(n):
                p = ()
                for j in range(len(Dict)):
                    if x[i] == Dict[1][j]:
                        p+=tuple(Dict[0][j])
                self.tableau[0][i] = p
            
            for i in range(1,n):
                for j in range(0,n-i):
                    p = ()
                    for k in range(0,i):
                        q = itertools.product(self.tableau[k][j],self.tableau[i-1-k][j+k+1])
                        for item in q:
                            x= ''.join(item)
                            for k in range(len(Dict)):
                                if x == Dict[1][k]:
                                    p+=tuple(Dict[0][k])
                    self.tableau[i][j] = p
            
            
            return 'S' in self.tableau[n-1][0]
                
            
p = CYK(Dict)

def unit_test_1():
    return p.isINCFL("abc")

def unit_test_2():
    return p.isINCFL("abbbabb")

def unit_test_3():
    return p.isINCFL("abbc")

def unit_test_4():
    return p.isINCFL("bbc")

def unit_test_5():
    return p.isINCFL("aaabb")

print(unit_test_1())
print(unit_test_2())
print(unit_test_3())
print(unit_test_4())
print(unit_test_5())


           