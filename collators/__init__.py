from collators.list_collator import collate as list_collator


def get_collator(name):
    return globals()[name]



