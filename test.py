# food was good not great not worth the wait or another visit	0,1 FOOD#QUALITY 1 2,5	-1,-1 RESTAURANT#GENERAL 0 5,7
# Food was good not great not worth the wait or another visit####[['Food', 'food quality', 'neutral', 'good not great not worth the wait or another visit']]

def transfom_data(data):
    sentence = data.split('####')[0].lower()
    quad_arr = data.split('####')[1].lower()
    cleaned_string = eval(quad_arr)
    result = sentence
    sen_map = {0:''}
    for i,quad in enumerate(cleaned_string):
        aspect = quad[0]
        category = quad[1]
        polarity = quad[2]
        opinion = quad[3]
        # 处理aspect，opinion，找到对应的索引下标
        a1,a2 = find_word_substring_positions(sentence,aspect)
        o1,o2 = find_word_substring_positions(sentence,opinion)
        c = category.upper().replace(' ','#')
        if polarity == 'positive':
            p=2
        elif polarity == 'negative':
            p=0
        elif polarity == 'neutral':
            p=1
        # print(a1,a2,o1,o2,c,p)
        result += '\t' + str(a1) + ',' +str(a2) + ' ' + c + ' ' + str(p) + ' ' + str(o1) + ',' + str(o2)
    return result
def find_word_substring_positions(string, substring):
    words = string.split()
    subwords = substring.split()
    start_index = None
    end_index = None
    for i in range(len(words)):
        if words[i:i+len(subwords)] == subwords:
            start_index = i
            end_index = i + len(subwords)
            break
    if start_index is None:
        return None, None  # 如果子串不存在于字符串中，返回 None
    return start_index, end_index


input_file = "input.txt"
output_file = "output.txt"
with open(input_file, "r") as f_input, open(output_file, "w") as f_output:
    # 逐行读取输入文件，并处理每一行
    for line in f_input:
        # 处理每一行字符串
        processed_line = transfom_data(line)
        # 将处理结果写入到输出文件的一行中
        f_output.write(processed_line + "\n")