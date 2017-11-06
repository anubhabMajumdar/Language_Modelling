fileName = "JuliusCaesar.txt"
f = open(fileName, 'r')
lines = f.readlines()
lines = [x for x in data if len(x)>1]

data = ""
for i in lines:
	s+=i