import g4f

print([
    provider.__name__
    for provider in g4f.Provider.__providers__
    if provider.working
])