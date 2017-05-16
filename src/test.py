import subprocess

var = "likely being the best"
source = __file__.split("/")
if len(source) > 1:
	path = source[0]
	if len(source) > 2:
		for n in range(len(source)-2):
			path = path + "/"+source[n+1]
	path = path+"/../../stemmer.pl"
else:
	path = "../../stemmer.pl"
t = subprocess.Popen(["perl", path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
test = t.communicate(input=var)
print(test[0].strip())
