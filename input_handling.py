def get_confirmation(prompt):
    while True:
        value = input(prompt + ' (Y/N): ')
        if value == 'Y':
            return True
        elif value == 'N':
            return False
        else:
            print(' <Invalid choice>')


def get_verified_input(prompt, allowed_values):
    while True:
        value = input(prompt)
        if value in allowed_values:
            return value
        else:
            print(' <Invalid choice>')


def get_verified_int_input(prompt, only_positive=False):
    while True:
        try:
            value = int(input(prompt))
            if only_positive and value < 0:
                print(' <Chosen value must be positive>')
                continue

        except ValueError:
            print(' <Invalid choice>')
            continue

        return value


def get_verified_float_input(prompt, only_positive=False):
    while True:
        try:
            value = float(input(prompt))
            if only_positive and value < 0:
                print(' <Chosen value must be positive>')
                continue

        except ValueError:
            print(' <Invalid choice>')
            continue

        return value


def get_verified_string_input(prompt):
    while True:
        try:
            value = str(input(prompt))
        except ValueError:
            print(' <Invalid choice>')
            continue

        return value
