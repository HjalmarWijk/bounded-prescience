from prescience.labelling.properties import prop_map
import sys

if __name__ == '__main__':
    if len(sys.argv)<=1:
        for env in prop_map:
            print(env)
    elif sys.argv[1]=='--repeat':
        for env in prop_map:
            for prop in prop_map[env]:
                print(env)
    elif sys.argv[1]=='--prop':
        for env in prop_map:
            for prop in prop_map[env]:
                print(prop)
         
            

