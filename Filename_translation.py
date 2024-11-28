import os
import shutil
import matplotlib.pyplot as plt
# 原始文件夹路径
source_folder = r'D:\Desktop\data_NoSeg\Day5'

# 新文件夹路径
destination_folder = r'D:\Desktop\data_NoSeg\Day5_result'

# 确保目标文件夹存在，如果不存在则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历原始文件夹中的文件
# Execellent_range = 0
Good_range = 0
Average_range = 0
Poor_range = 0
# Fail_range = 0
for filename in os.listdir(source_folder):
    # 检查文件是否是图像文件（这里假设只处理.jpg和.png文件，可以根据需要修改）
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
        # 构造原始文件的完整路径
        source_file = os.path.join(source_folder, filename)
        # 找到"DAY3"后面的字符串
        # start_index = filename.find("DAY3") or filename.find("DAY5")

        # 提取子字符串
        start_index_day3 = filename.find("DAY3")
        start_index_day5 = filename.find("DAY5")
        # 找到".JPG"前面的字符串
        end_index = filename.find(".JPG")
        if start_index_day3 != -1:
            # 找到了 "DAY3"
            start_index = start_index_day3
            extracted_string = filename[start_index + 5:end_index]
            # if extracted_string in ['13C1', '12C1', '11C1', '10C1', '9C1', '8C1', '7C1', '6C1']:
            #     range = 'Excellent'
            #     Execellent_range += 1
            # elif extracted_string in ['13C2', '12C2', '11C2', '10C2', '9C2', '8C2', '7C2', '6C2', '5C1', '4C1']:
            #     range = 'Good'
            #     Good_range += 1
            if extracted_string in ['13C1', '12C1', '11C1', '10C1', '9C1', '8C1', '7C1', '6C1',
                                    '13C2', '12C2', '11C2', '10C2', '9C2', '8C2', '7C2', '6C2', '5C1', '4C1']:
                range = 'Good'
                Good_range += 1
            elif extracted_string in ['13C3', '12C3', '11C3', '10C3', '9C3', '8C3', '7C3', '6C3', '5C2', '4C2', '3C2',
                                      '3C1']:
                range = 'Average'
                Average_range += 1
            elif extracted_string in ['5C3', '4C3', '3C3', '2C2', '2C1']:
                range = 'Poor'
                Poor_range += 1
            else:
                range = 'Poor'
                Poor_range += 1
            # else:
            #     range = 'Fail'
            #     Fail_range += 1

        elif start_index_day5 != -1:
            # 找到了 "DAY5"
            start_index = start_index_day5
            extracted_string = filename[start_index + 5:end_index]
            # print(extracted_string)
            # if extracted_string in ['6AA', '5AA', '4AA', '3AA']:
            #     range = 'Excellent'
            #     Execellent_range += 1
            # elif extracted_string in ['6AB', '5AB', '4AB', '3AB', '6BA', '5BA', '4BA', '3BA']:
            #     range = 'Good'
            #     Good_range += 1
            if extracted_string in ['6AA', '5AA', '4AA', '3AA', '6AB', '5AB', '4AB', '3AB', '6BA', '5BA', '4BA', '3BA']:
                range = 'Good'
                Good_range += 1
            elif extracted_string in ['6BB', '5BB', '4BB', '3BB', '6AC', '5AC', '4AC', '3AC', '6CA', '5CA',
                                      '4CA', '3CA']:
                range = 'Average'
                Average_range += 1
            elif extracted_string in ['LM']:
                range = 'Poor'
                Poor_range += 1
            else:
                range = 'Poor'
                Poor_range += 1
            # else:
            #     range = 'Poor'
            #     Poor_range += 1

        # 根据需要对文件进行重命名
        # 这里假设你想要将所有文件重命名为'new_prefix_xxx.jpg'，其中xxx是文件的原始名称
        new_filename = filename[0:start_index + 4] + '_' + str(range) + '.jpg'

        # 构造目标文件的完整路径
        destination_file = os.path.join(destination_folder, new_filename)

        # 将文件从原始位置复制到目标位置并重命名
        shutil.copy(source_file, destination_file)

        # 如果需要，可以删除原始文件
        # os.remove(source_file)

# print("Execellent:", Execellent_range, "Good:", Good_range, "Average:", Average_range, "Poor:", Poor_range, "Fail:",
#       Fail_range)

print("Good:", Good_range, "Average:", Average_range, "Poor:", Poor_range, "Total:", Good_range + Average_range
      + Poor_range)

# Create a dictionary to store category statistics
category_statistics = {
    # 'Excellent': Execellent_range,
    'Good': Good_range,
    'Average': Average_range,
    'Poor': Poor_range
    # 'Fail': Fail_range
}

# Create a bar chart
plt.bar(category_statistics.keys(), category_statistics.values())

# Add labels to the chart
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Statistics')

# Add numbers on top of the bars
for category, count in category_statistics.items():
    plt.text(category, count, str(count), ha='center', va='bottom')

if start_index_day3 != -1:
    plt.title('Day3')
    plt.savefig(destination_folder + '/' + 'day3_category_statistics.png')

elif start_index_day5 != -1:
    plt.title('Day5')
    plt.savefig(destination_folder + '/' + 'day5_category_statistics.png')


# Display the chart
plt.show()
