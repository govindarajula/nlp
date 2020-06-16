import unicodedata


def pre_process(data):
    for idx, item in enumerate(data):
        data[idx] = unicodedata.normalize("NFKD", item)
        data[idx] = ''.join(c for c in unicodedata.normalize('NFD', item)
                  if unicodedata.category(c) != 'Mn')
    return data

print(['the\u00a0years'])
print(pre_process(['the\u00a0years','the\u00a0years']))