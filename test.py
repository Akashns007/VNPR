def words_to_rupees(s):
    num_dict = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, 
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, 
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, 
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, 
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "paise": 0.0
    }

    ignore_words = {"rupees", "and"}  # Ignoring unnecessary words
    valid_words = [word for word in s if word not in ignore_words]

    total = 0
    cur_sum = 0

    for ch in valid_words:
        num = num_dict[ch]

        if num in (100, 1000, 100000):
            cur_sum = cur_sum * num
            if num in (1000, 100000):
                total += cur_sum
                cur_sum = 0
        else:
            cur_sum += num

    return total + cur_sum

def divide_paise(s):
    sub_string = []
    for i in range(len(s) - 1, 0, -1):
        if s[i] == "and":
            sub_string = s[i+1:] 
            return s[0:i], sub_string
    return sub_string, s
        
def merge(main_amt, paise_amt):
    paise_amt = paise_amt/100 
    return main_amt + paise_amt
    
    
# 🏆 **Example Usage:**
inp_text = input("Enter a number in words: ")
# inp_text = "four lakhs twenty thousand seven hundred twenty eight and five paise only"
# inp_text = "ninety five paise"
# inp_text = "nine hundred and ninety five paise only"
# inp_text = "four hundred and twenty thousand seven hundred twenty eight"

s = inp_text.lower().strip().split()

if s[-1] == "only":
    s = s[:-1]
if s[-1] == "paise":
    s1, s2 = divide_paise(s[:-1])
    main_amt = words_to_rupees(s1)
    paise_amt = words_to_rupees(s2)
    converted_amt = merge(main_amt, paise_amt)
else:
    converted_amt = words_to_rupees(s)
    
print("Converted Rupees:", converted_amt)
