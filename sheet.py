def _get_layers(cls, num_vertices:int, target_size:int = 3):
    if target_size%3 > 0: raise ValueError('target must be a multiple of 3')

    if num_vertices ==3:
        return [(3,3)]
    
    linear_layers = []
    remaining = num_vertices*3
    remaining = num_vertices*3
    while remaining !=target_size:
        start_value = remaining
        end_value = remaining -3
        linear_layers.append((start_value, end_value))
        remaining-=3

    return linear_layers


print(_get_layers(None, 8,target_size=9))