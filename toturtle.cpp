// by NAKUL AGGARWAL (19CS10044)

#include <bits/stdc++.h>
#include <iostream>
#include <iosfwd>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <tuple>
#include <utility>
#include <queue>
using namespace std;
#define vec vector
#define range(i,x,y)    for(int i=x; i<=y; i++)
#define ranger(i,x,y)   for(int i=x; i>=y; i--)

map<string, map<string, vec<string>>> triples;
map<string, string> types;

void read_triple ( string & s , string & p , string & o ) {
    string line;
    getline(cin, line);

    if ( line == "-1" ) {
        s = line;
        return;
    }

    int i = 0;
    int len = line.size();
    vec<string> triple;

    while ( i < len ) {
        string cell;
        if ( line[i] == '<' ) {
            i ++;
            while ( line[i] != '>' )
                cell.push_back(line[i++]);
            i += 2;
        }

        else if ( line[i] == '"' ) {
            i ++;
            while ( line[i] != '"' )
                cell.push_back(line[i++]);
            i += 2;
        }

        else {
            while ( i < len && line[i] != ' ' )
                cell.push_back(line[i++]);
            i ++;
        }

        triple.push_back(cell);
    }

    s = triple[0];
    p = triple[1];
    o = triple[2];

    string restr = " ,;:.";
    for ( char c : restr ) {
        replace(s.begin(), s.end(), c, '_');
        replace(p.begin(), p.end(), c, '_');
    }
}

void insert_triple ( const string & s , const string & p , const string & o ) {
    if ( triples.find(s) == triples.end() )
        triples.insert({s, {}});
    
    auto & mm = triples.at(s);
    if ( mm.find(p) == mm.end() )
        mm.insert({p, {}});
    
    mm.at(p).push_back(o);
}

void takein ( ) {
    string s, p, o;
    
    read_triple(s, p, o);
    while ( s != "-1" ) {
        insert_triple(s, p, o);
        read_triple(s, p, o);
    }

    string line;
    getline(cin, line);
    while ( line != "-1" )
        getline(cin, line);
    
    string pred, type;
    cin >> pred;
    while ( pred != "-1" ) {
        cin >> type;
        types.insert({pred, type});
        cin >> pred;
    }
}

string getterminator ( int pnum , int onum ) {
    if ( pnum == 1 && onum == 1 )   return " .\n";
    if ( onum == 1 )    return " ;\n";
    return ",";
}

bool isblanknode ( const string & s ) {
    if ( s.size() < 2 ) return 0;
    return (s[0] == '_' && s[1] == ':');
}

string typedobject ( const string & o , const string & p ) {
    if ( p == "type" )    return "exs:" + o;
    if ( isblanknode(o) )   return o;
   
    auto type = types.at(p);
    if ( type == "entity" ) {
        string o2 = o;
        string restr = " ,;:.";
        for ( char c : restr ) 
            replace(o2.begin(), o2.end(), c, '_');
        return "exr:" + o2;
    }
    
    if ( type == "string" ) return "\"" + o + "\"^^xsd:string";
    if ( type == "float" ) return "\"" + o + "\"^^xsd:float";
    if ( type == "bool" ) return "\"" + o + "\"^^xsd:boolean";
    if ( type == "int" ) return "\"" + o + "\"^^xsd:int";
    return "\"" + o + "\"^^xsd:date";
}

void print ( ) {
    cout << "\n";
    cout << "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>.\n";
    cout << "@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.\n";
    cout << "@prefix exs: <http://example.org/schema#>.\n";
    cout << "@prefix exr: <http://example.org#>.\n\n";

    for ( auto & s_po : triples ) {
        auto & s = s_po.first;
        int pnum = s_po.second.size();

        if ( isblanknode(s) )   cout << s;
        else    cout << "exr:" << s;

        for ( auto & p_o : s_po.second ) {
            if ( pnum < s_po.second.size() )
                cout << '\t';
            else    cout << ' ';
            
            auto & p = p_o.first;
            int onum = p_o.second.size();

            if ( p == "type" )  cout << "rdf:" << p;
            else    cout << "exs:" << p;

            for ( auto & o : p_o.second ) {
                cout << ' ' << typedobject(o, p);
                cout << getterminator(pnum, onum);
                onum --;
            }

            pnum --;
        }
    }

    cout << "\n";
}


int main ( ) {
    takein();
    print();
}
