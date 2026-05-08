
text = "0"

operand = "0"
total = 0

nextOp = 0
errorState = False


def Reset():
    text.text = "0"
    operand = "0"
    total = 0
    nextOp = 0
    errorState = False

def AddChar(ch):
    if (not errorState):
        operand += ch
        text = operand
 
def Operation(op):
    if (not errorState):
        try:
            eq = float(operand)
            Process(eq)
            nextOp = op
        except:
             pass
    else:
        text.text = "ERROR"
        errorState = True

def Process(eq):
    if(nextOp == 1):
        total += eq
    elif(nextOp == 2):
        total -= eq
    elif(nextOp == 3):
            total *= eq
    elif(nextOp == 4):
            total /= eq
    elif(nextOp == 0):
            total = eq


