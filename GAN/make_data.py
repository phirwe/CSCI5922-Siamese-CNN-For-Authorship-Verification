import os
authors = ['085', '037', '118', '107', '026', '125']

datadir = '../Dataset/Authors'

# Authors has a list of all the authors
# Now create train - test pairs for each (assuming 50 total in each, 0.8*50 = 40 for train and 0.2*50 = 10 for test)
train_count = 0
test_count = 0

for authorA in authors:
    trainA_filenames = []
    testA_filenames = []
    for x in os.listdir(datadir + '/' + authorA):
        if x != '.DS_Store':
            if len(trainA_filenames) < 40:
                trainA_filenames.append(x)
            elif len(testA_filenames) < 10:
                testA_filenames.append(x)
            else:
                break 
    print (len(trainA_filenames), len(testA_filenames))
    
    for authorB in authors:
        if authorA == authorB:
            continue
        print (authorA, 'to', authorB)
        rootdir = './datasets/' + authorA + 'to' + authorB
        trainA_dir = rootdir + '/trainA'
        trainB_dir = rootdir + '/trainB'
        testA_dir = rootdir + '/testA'
        testB_dir = rootdir + '/testB'
        
        # Comment these lines out if directories already exist
        os.system('mkdir ' + rootdir)
        os.system('mkdir ' + trainA_dir)
        os.system('mkdir ' + trainB_dir)
        os.system('mkdir ' + testA_dir)
        os.system('mkdir ' + testB_dir)
        trainB_filenames = []
        testB_filenames = []
        for x in os.listdir(datadir + '/' + authorB):
            if x != '.DS_Store':
                if len(trainB_filenames) < 40:
                    trainB_filenames.append(x)
                elif len(testB_filenames) < 10:
                    testB_filenames.append(x)
                else:
                    break
                    
        print (len(trainB_filenames), len(testB_filenames))
        
        for i in range(40):
            nameA = trainA_filenames[i]
            nameB = trainB_filenames[i]
            os.system('cp ' + datadir + authorA + '/' + nameA + ' ' + trainA_dir + '/' + nameA)
            os.system('cp ' + datadir + authorB + '/' + nameB + ' ' + trainB_dir + '/' + nameB)
            if i < 10:
                testA = testA_filenames[i]
                testB = testB_filenames[i]
                os.system('cp ' + datadir + authorA + '/' + testA + ' ' + testA_dir + '/' + testA)
                os.system('cp ' + datadir + authorB + '/' + testB + ' ' + testB_dir + '/' + testB)
