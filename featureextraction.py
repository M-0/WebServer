import math
import numpy
import numpy as np
import pickle

class extractFeatures():
    def seqToMat(seq):
        encoder = ['a','c','t','g']
        len = seq.__len__()
        n = int(math.ceil(math.sqrt(len)))
        seqMat = [[0 for x in range(n)] for y in range(n)]
        i = 0
        seqiter = 0
        for i in range(n):
            j = 0
            for j in range(n):
                if seqiter < len:
                    try:
                        aa = int(encoder.index(seq[seqiter]))
                    except ValueError:
                        exit(0)
                    else:
                        seqMat[i][j] = aa
                    seqiter += 1
        return seqMat

                                                               ###frequencyVector###
    def frequencyVec4(seq):
        encoder = ['a','c','t','g']
        fv = [0 for x in range(4)]
        i = 0
        for i in range(4):
            fv[i] = seq.count(encoder[i])
        #print('FV')
        #print(fv)    
        return fv

    def frequencyVec16(encoding1,seq):
        fv = [0 for x in range(16)]
        i = 0
        for i in range(16):
            fv[i] = encoding1.count(i)
        #print('FV')
        #print(fv) 
        return fv

    def frequencyVec64(encoding2,seq):
        fv = [0 for x in range(64)]
        i = 0
        for i in range(64):
            fv[i] = encoding2.count(i)
        #print('FV')
        #print(fv) 
        return fv


                                                              ###AAPIV Matrices###

    def AAPIV4(seq):
        encoder = ['a','c','t','g']
        apv = [0 for x in range(4)]
        i = 1
        sum = 0
        for i in range(4):
            j = 0
            for j in range(len(seq)):
                if seq[j] == encoder[i]:
                    sum = sum + j + 1
            apv[i] = sum
            sum = 0
        #print('AAPIV')   
        #print(apv[1:] + apv[0:1])
        return apv[1:] + apv[0:1]

    def AAPIV16(encoding1,seq):
        apv = [0 for x in range(16)]
        i = 1
        sum = 0
        for i in range(16):
            j = 0
            for j in range(len(encoding1)):
                if encoding1[j] == i:
                    sum = sum + j + 1
            apv[i] = sum
            sum = 0
        #print('AAPIV') 
        #print(apv[1:] + apv[0:1])
        return apv[1:] + apv[0:1]

    def AAPIV64(encoding2,seq):
        apv = [0 for x in range(64)]
        i = 1
        sum = 0
        for i in range(64):
            j = 0
            for j in range(len(encoding2)):
                if encoding2[j] == i:
                    sum = sum + j + 1
            apv[i] = sum
            sum = 0
        #print('AAPIV')
        #print(apv[1:] + apv[0:1])
        return apv[1:] + apv[0:1]

    def print2Dmat(mat):
        n = len(mat)
        i = 0
        strOut = ''
        for i in range(n):
            strOut = strOut + str(mat[i]) + '<br>'
        return strOut
                                                       ####Prim matrices###

    def PRIM4(seq):
        encoder = ['a','c','t','g']
        prim = [[0 for x in range(4)] for y in range(4)]
        i = 0
        for i in range(4):
            aa1 = encoder[i]
            aa1index = -1
            for x in range(len(seq)):
                if seq[x] == aa1:
                    aa1index = x + 1
                    break
            if aa1index != -1:
                j = 0
                for j in range(4):
                    if j != i:
                        aa2 = encoder[j]
                        aa2index = 0
                        for y in range(len(seq)):
                            if seq[y] == aa2:
                                aa2index = aa2index + ((y + 1) - aa1index)
                        prim[i][j] = int(aa2index)
        #print('prim') 
        #print(prim)                
        return prim

    def cal_dibase_index(seq):
      dibase_index = {'aa':1,'ac':2,'at':3,'ag':4,'ca':5,'ct':6,'cc':7,'cg':8,'gg':9,'ga':10,'gc':11,'gt':12,'tt':13,'ta':14,'tc':15,'tg':16} 
      return dibase_index[ seq]

    def PRIM16(encoding1):
        #print(encoding1)
        prim = [[0 for x in range(16)] for y in range(16)]
        i = 0
        #print(np.array(prim).shape)
        for i in range(len(prim)):
            #print('--------------------------------------')
            #print(encoding1[i],(i+1))
            #print('--------------------------------------')
            f = -1
            for j in range(len(encoding1)):
                #print(encoding1[j],(j+1))
                if (i+1) == encoding1[j] and f == -1:
                    f = j
                  #  print('i',f,encoding1[f])
                    break
            for j in range(len(encoding1)):
                if f != -1  and f != j:
    #                if i == 4 and encoding1[j] == 5:
    #                    print(j+1,encoding1[j],f+1,(j+1)-(f+1)-0)
                 #   print(encoding1[f]) 
                    if encoding1[f] == (i+1):
                        prim[i][encoding1[j]-1] += (j)-(f)
    #               break
        #print('prim')
        #print(prim)
        return prim

    def first_index(encoding1):
        #print(encoding1)
        prim = [[0 for x in range(16)] for y in range(16)]
        i = 0
        #print(np.array(prim).shape)
        results = np.array([0]*len(prim))
        for i in range(len(prim)):
            #print('--------------------------------------')
            #print(encoding1[i],(i+1))
            #print('--------------------------------------')
            for j in range(len(encoding1)):
                #print(encoding1[j],(j+1))
                if (i+1) == encoding1[j]:
                    results[i] = j+1
                  #  print('i',f,encoding1[f])
                    break

        return results
      
     
    def cal_tribase_index(seq):
     
      if seq[0] == 'a':
        x = extractFeatures.cal_dibase_index(seq[1]+seq[2])
        return(x+0)
      elif seq[0] == 'c':
        x = extractFeatures.cal_dibase_index(seq[1]+seq[2])
        return(x+16)
      elif seq[0] == 'g':
        x = extractFeatures.cal_dibase_index(seq[1]+seq[2])
        return(x+32)
      elif seq[0] ==  't':
        x = extractFeatures.cal_dibase_index(seq[1]+seq[2])
        return(x+48) 

    def PRIM64(encoding2):
        #print(encoding2)
        prim = [[0 for x in range(64)] for y in range(64)]
        i = 0
        #print(np.array(prim).shape)
        for i in range(len(prim)):
            #print('--------------------------------------')
            #print(encoding2[i],(i+1))
            #print('--------------------------------------')
            f = -1
            for j in range(len(encoding2)):
                #print(encoding2[j],(j+1))
                if (i+1) == encoding2[j] and f == -1:
                    f = j
                  #  print('i',f,encoding2[f])
                    break
            for j in range(len(encoding2)):
                if f != -1  and f != j:
    #                if i == 4 and encoding2[j] == 5:
    #                    print(j+1,encoding2[j],f+1,(j+1)-(f+1)-0)
                 #   print(encoding2[f]) 
                    if encoding2[f] == (i+1):
                        prim[i][encoding2[j]-1] += (j)-(f)
    #               break
        #print('prim')
        #print(prim)
        return prim
        

    def first_index(encoding2):
        #print(encoding2)
        prim = [[0 for x in range(64)] for y in range(64)]
        i = 0
        #print(np.array(prim).shape)
        results = np.array([0]*len(prim))
        for i in range(len(prim)):
            #print('--------------------------------------')
            #print(encoding2[i],(i+1))
            #print('--------------------------------------')
            for j in range(len(encoding2)):
                #print(encoding[j],(j+1))
                if (i+1) == encoding2[j]:
                    results[i] = j+1
                  #print('i',f,encoding2[f])
                    break
        return results



    def rawMoments(mat, order):
        n = len(mat)
        rawM = []
        sum = 0
        i = 0
        for i in range(order + 1):
            j = 0
            for j in range(order + 1):
                if i + j <= order:
                    p = 0
                    for p in range(n):
                        q = 0
                        for q in range(n):
                            sum = sum + (((p + 1) ** i) * ((q + 1) ** j) * int(mat[p][q]))
                    rawM.append(sum)
                    sum = 0
        #print('rawmmoments')            
        #print(rawM)
        return rawM


    def centralMoments(mat, order, xbar, ybar):
        n = len(mat)
        centM = []
        sum = 0
        i = 0
        for i in range(order + 1):
            j = 0
            for j in range(order + 1):
                if i + j <= order:
                    p = 0
                    for p in range(n):
                        q = 0
                        for q in range(n):
                            sum = sum + ((((p + 1) - xbar) ** i) * (((q + 1) - ybar) ** j) * mat[p][q])
                    centM.append(sum)
                    sum = 0
        #print('central moments')
        #print(centM)
        return centM


    def hahnMoments(mat, order):
        N = len(mat)
        hahnM = []
        i = 0
        for i in range(order + 1):
            j = 0
            for j in range(order + 1):
                if i + j <= order:
                    answer = extractFeatures.hahnMoment(i, j, N, mat)
                    hahnM.append(answer)
        #print('hahn moments')
        #print(hahnM)
        return hahnM


    def hahnMoment(m, n, N, mat):
        value = 0.0
        x = 0
        for x in range(N):
            y = 0
            for y in range(N):
                value = value + (
                            mat[x][y] * (extractFeatures.hahnProcessor(x, m, N)) * (extractFeatures.hahnProcessor(x, n, N)))
        #print('value')
        #print(value)
        return value


    def hahnProcessor(x, n, N):
        return extractFeatures.hahnPol(x, n, N) * math.sqrt(extractFeatures.roho(x, n, N))


    def hahnPol(x, n, N):
        answer = 0.0
        ans1 = extractFeatures.pochHammer(N - 1.0, n) * extractFeatures.pochHammer(N - 1.0, n)
        ans2 = 0.0
        k = 0
        for k in range(n + 1):
            ans2 = ans2 + math.pow(-1.0, k) * ((extractFeatures.pochHammer(-n, k) * extractFeatures.pochHammer(-x, k) *
                                                extractFeatures.pochHammer(2 * N - n - 1.0, k)))
        answer = ans1 + ans2
        return answer


    def roho(x, n, N):
        return extractFeatures.gamma(n + 1.0) * extractFeatures.gamma(n + 1.0) * extractFeatures.pochHammer((n + 1.0), N)


    def gamma(x):
        return math.exp(extractFeatures.logGamma(x))


    def logGamma(x):
        temp = (x - 0.5) * math.log(x + 4.5) - (x + 4.5)
        ser = 101.19539853003
        return temp + math.log(ser * math.sqrt(2 * math.pi))


    def pochHammer(a, k):
        answer = 1.0
        i = 0
        for i in range(k):
            answer = answer * (a + i)
        return answer


    def calcFV(seq):
        encoding1 = list()
        for i in range(0,len(seq)-1,2):
            encoding1.append(extractFeatures.cal_dibase_index(seq[i]+seq[i+1]))
            
        encoding2 = list()
        for i in range(0,len(seq)-2,3):
            encoding2.append(extractFeatures.cal_tribase_index(seq[i]+seq[i+1]+seq[i+2]))

        fv = [0 for x in range(450)]
        fvIter = 0
        myMat = extractFeatures.seqToMat(seq)
        myRawMoments = extractFeatures.rawMoments(myMat, 3)
        for ele in myRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        xbar = myRawMoments[4]
        ybar = myRawMoments[1]
        myCentralMoments = extractFeatures.centralMoments(myMat, 3, xbar, ybar)
        for ele in myCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myHahnMoments = extractFeatures.hahnMoments(myMat, 3)
        for ele in myHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myFrequencyVec4 = extractFeatures.frequencyVec4(seq)
        for ele in myFrequencyVec4:
            fv[fvIter] = ele
            fvIter = fvIter + 1

            fvIter = 0

        myMat = extractFeatures.seqToMat(seq)
        myRawMoments = extractFeatures.rawMoments(myMat, 3)
        for ele in myRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        xbar = myRawMoments[4]
        ybar = myRawMoments[1]
        myCentralMoments = extractFeatures.centralMoments(myMat, 3, xbar, ybar)
        for ele in myCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myHahnMoments = extractFeatures.hahnMoments(myMat, 3)
        for ele in myHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myFrequencyVec16 = extractFeatures.frequencyVec16(encoding1,seq)
        for ele in myFrequencyVec16:
            fv[fvIter] = ele
            fvIter = fvIter + 1

            fvIter = 0
        myMat = extractFeatures.seqToMat(seq)
        myRawMoments = extractFeatures.rawMoments(myMat, 3)
        for ele in myRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        xbar = myRawMoments[4]
        ybar = myRawMoments[1]
        myCentralMoments = extractFeatures.centralMoments(myMat, 3, xbar, ybar)
        for ele in myCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myHahnMoments = extractFeatures.hahnMoments(myMat, 3)
        for ele in myHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myFrequencyVec64 = extractFeatures.frequencyVec64(encoding2,seq)
        for ele in myFrequencyVec64:
            fv[fvIter] = ele
            fvIter = fvIter + 1
            
            
     
        myPRIM = extractFeatures.PRIM4(seq)
        myPRIMRawMoments = extractFeatures.rawMoments(myPRIM, 3)
        xbar2 = myPRIMRawMoments[4]
        ybar2 = myPRIMRawMoments[1]
        myPRIMCentralMoments = extractFeatures.centralMoments(myPRIM, 3, xbar2, ybar2)
        for ele in myPRIMRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        for ele in myPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myPRIMHahnMoments = extractFeatures.hahnMoments(myPRIM, 3)
        for ele in myPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
            
            myPRIM = extractFeatures.PRIM16(encoding1)
        myPRIMRawMoments = extractFeatures.rawMoments(myPRIM, 3)
        xbar2 = myPRIMRawMoments[4]
        ybar2 = myPRIMRawMoments[1]
        myPRIMCentralMoments = extractFeatures.centralMoments(myPRIM, 3, xbar2, ybar2)
        for ele in myPRIMRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        for ele in myPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myPRIMHahnMoments = extractFeatures.hahnMoments(myPRIM, 3)
        for ele in myPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1

            myPRIM = extractFeatures.PRIM64(encoding2)
        myPRIMRawMoments = extractFeatures.rawMoments(myPRIM, 3)
        xbar2 = myPRIMRawMoments[4]
        ybar2 = myPRIMRawMoments[1]
        myPRIMCentralMoments = extractFeatures.centralMoments(myPRIM, 3, xbar2, ybar2)
        for ele in myPRIMRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        for ele in myPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myPRIMHahnMoments = extractFeatures.hahnMoments(myPRIM, 3)
        for ele in myPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1

        myAAPIV = extractFeatures.AAPIV4(seq)
        for ele in myAAPIV:
            fv[fvIter] = ele
            fvIter = fvIter + 1

            myAAPIV = extractFeatures.AAPIV16(encoding1,seq)
        for ele in myAAPIV:
            fv[fvIter] = ele
            fvIter = fvIter + 1

            myAAPIV = extractFeatures.AAPIV64(encoding2,seq)
        for ele in myAAPIV:
            fv[fvIter] = ele
            fvIter = fvIter + 1

        myRPRIM = extractFeatures.PRIM4(seq[::-1])
        myRPRIMRawMoments = extractFeatures.rawMoments(myRPRIM, 3)
        xbar3 = myRPRIMRawMoments[4]
        ybar3 = myRPRIMRawMoments[1]
        myRPRIMCentralMoments = extractFeatures.centralMoments(myRPRIM, 3, xbar3, ybar3)
        for ele in myRPRIMRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        for ele in myRPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myRPRIMHahnMoments = extractFeatures.hahnMoments(myRPRIM, 3)
        for ele in myRPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1

        myRPRIM = extractFeatures.PRIM16(encoding1[::-1])
        myRPRIMRawMoments = extractFeatures.rawMoments(myRPRIM, 3)
        xbar3 = myRPRIMRawMoments[4]
        ybar3 = myRPRIMRawMoments[1]
        myRPRIMCentralMoments = extractFeatures.centralMoments(myRPRIM, 3, xbar3, ybar3)
        for ele in myRPRIMRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        for ele in myRPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myRPRIMHahnMoments = extractFeatures.hahnMoments(myRPRIM, 3)
        for ele in myRPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
          
        myRPRIM = extractFeatures.PRIM64(encoding2[::-1])
        myRPRIMRawMoments = extractFeatures.rawMoments(myRPRIM, 3)
        xbar3 = myRPRIMRawMoments[4]
        ybar3 = myRPRIMRawMoments[1]
        myRPRIMCentralMoments = extractFeatures.centralMoments(myRPRIM, 3, xbar3, ybar3)
        for ele in myRPRIMRawMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        for ele in myRPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1
        myRPRIMHahnMoments = extractFeatures.hahnMoments(myRPRIM, 3)
        for ele in myRPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter = fvIter + 1

        myRAAPIV = extractFeatures.AAPIV4(seq[::-1])
        for ele in myRAAPIV:
            fv[fvIter] = ele
            fvIter = fvIter + 1
            

        myRAAPIV = extractFeatures.AAPIV16(encoding1,seq[::-1])
        for ele in myRAAPIV:
            fv[fvIter] = ele
            fvIter = fvIter + 1

        myRAAPIV = extractFeatures.AAPIV64(encoding2,seq[::-1])
        for ele in myRAAPIV:
            fv[fvIter] = ele
            fvIter = fvIter + 1

        return fv
    def processAllStrings(fname):
        seqs = []
        allFVs = []
        with open(fname, 'r') as filehandle:
            for line in filehandle:
                currentPlace = line[:-1]
                seqs.append(currentPlace)
        allowed_chars = set('agtc')
        i = 0
        for seq in seqs:
            seq=seq.lower()
            print(str(i)+': '+seq)
            if seq != '':
                if set(seq).issubset(allowed_chars):
                    #print(len(seq))
                    allFVs.append(calcFV(seq))
                    i = i + 1
                else:
                    print('Invalid Sequence\n' + str(i))
                    i = i + 1
        return allFVs

    def matrix(string, length):
        return (string[0 + i:length + i] for i in range(0, len(string), length))

    def feature_result(seq):
        if seq.isalpha():
            seq = seq[0:100]
            seq = seq.lower()
            #out = extractFeatures.performPrediction(numpy.asarray(extractFeatures.calcFV(seq)))
            seq = "".join(list(extractFeatures.matrix(seq, 100)))
#            return [seq, str(out[0]), out[1]]
            return numpy.asarray(extractFeatures.calcFV(seq))
        else:
            return ['Invalid Sequence']

    def decode_sigma(res):
        sequences = ['Sigma 24','Sigma 28','Sigma 32','Sigma 38','Sigma 54','Sigma 70']
        return sequences[np.argmax(res)]

    def scaling(FV):
        FV = std_scale.transform(FV)
        return FV
