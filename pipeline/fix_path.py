content = open("pipeline.py").read().replace("demo/sample.txt", "../demo/sample.txt")
open("pipeline.py", "w").write(content)
print("Done!")
