import pandas as pd

images = dict()

boxes = pd.read_csv('full_bboxes.csv', header=None)

for row in boxes.values:
    if row[0] not in images:
        images[row[0]] = []
    images[row[0]].append(row)

print(images.keys())
print(len(images))

i = 0
columns = ['path', 'x1', 'y1', 'x2', 'y2', 'class']
test_data = pd.DataFrame(columns=columns)
train_data = pd.DataFrame(columns=columns)
for image in images.keys():
    row = pd.DataFrame(images[image], columns=columns)
    if i < 7:
        train_data = train_data.append(row, ignore_index=True)
    else:
        test_data = test_data.append(row, ignore_index=True)
    i = (i + 1) % 10

train_data.to_csv('train.csv', header=None, index=False)
test_data.to_csv('test.csv', header=None, index=False)
