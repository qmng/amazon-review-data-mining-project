import subprocess

var = "likely being the best"
t = subprocess.Popen(["perl", "stemmer.pl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
test = t.communicate(input=var)
print(test[0].strip())
