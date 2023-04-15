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
#define endl    "\n"
#define triple_t    vec<int>
#define row_t       vec<string>
#define ENTITY  0
#define STRING  1
#define FLOAT   2
#define INT     3
#define BOOL    4
#define DATE    5
#define firstEnt    get<0>
#define secEnt      get<1>
#define featVec     get<2>
#define rfal_sample_t   tuple<string, string, vec<double>>
#define verbose 1


class Converter;
class DataBase;
class DecisionTree;
class KnowledgeGraph;
class LinguisticMatching;
class RandomForest;
class RandomForestActiveLearner;
class RecordLinker;
class SchemaMapper;
class Table;


map<string, int> TYPEID = {
    {"float", FLOAT},   {"bool", BOOL},
    {"string", STRING}, {"int", INT},
    {"entity", ENTITY}, {"date", DATE}
};
vec<string> TYPENAME = {"entity", "string", "float", "int", "bool", "date"};
// LARGER DATA-TYPES AT LOWER INDICES


int Random ( int x , int y ) {
    random_device device;
    mt19937 rand_num (device());
    return uniform_int_distribution<int>(x, y)(rand_num);
}


class LinguisticMatching {
    private:
        static char tolower ( char c ) {
            if ( c >= 'A' && c <= 'Z' )
                return 'a' + (c - 'A');
            return c;
        }

        static string tolower ( const string & s ) {
            string p;
            for ( char c : s )  p.push_back(tolower(c));
            return p;
        }

        static int LCS ( const string & s1, const string & s2 ) {
            int m = s1.size(), n = s2.size();
            vec<vec<int>> lcs (m+1, vec<int>(n+1, 0));
            int len = 0;
            ranger(i, m-1, 0)
                ranger(j, n-1, 0) {
                    lcs[i][j] = 0;
                    if ( s1[i] == s2[j] )
                        lcs[i][j] = 1 + lcs[i+1][j+1];
                    len = max<int>(len, lcs[i][j]);
                }
            return len;
        }

        static bool isalphabet ( char c ) {
            if ( c <= 'z' && c >= 'a' ) return 1;
            if ( c <= 'Z' && c >= 'A' ) return 1;
            return 0;
        }

        static string soundex ( string s ) {
            vec<string> codes = {"aeiou", "bfpv", "cgjkqsxz", "dt", "l", "mn", "r"};
            string toremove = "hyw";
            map<char, char> charcodes;
            range(i, 0, 6)
                for ( char c : codes[i] )
                    charcodes.insert({c, i+'0'});
            
            s = tolower(s);
            string filter1;
            for ( char c : s ) {
                if ( isalphabet(c) && toremove.find_first_of(c) == string::npos )
                    filter1.push_back(charcodes.at(c));
            }

            if ( ! filter1.size() )   return "";

            string filter2;
            char prevchar = 0;
            for ( char c : filter1 ) {
                if ( c != prevchar ) {
                    filter2.push_back(c);
                    prevchar = c;
                }
            }

            string filter3;
            for ( char c : filter2 )
                if ( c != '0' ) filter3.push_back(c);
            
            while ( filter3.size() < 4 )
                filter3.push_back('0');

            return filter3.substr(0, 4);
        }

        static double vecL2Norm ( const vec<int> & v ) {
            int s = 0;
            for ( int x : v )   s += x * x;
            return sqrtl(s);
        }

    public:
        static string shortform ( const string & s ) {
            string p;
            bool picknext = 1;
            for ( char c : s ) {
                if ( ! isalphabet(c) ) picknext = 1;
                else if ( picknext ) {
                    p.push_back(tolower(c));
                    picknext = 0;
                }
            }
            return p;
        }

        static int EditDistance ( const string & S1, const string & S2 ) {
            auto s1 = tolower(S1), s2 = tolower(S2);
            int m = s1.size(), n = s2.size();
            vec<vec<int>> ed (m+1, vec<int>(n+1));

            range(i, 0, n)  ed[m][i] = n-i;
            range(j, 0, m)  ed[j][n] = m-j;

            ranger(i, m-1, 0)
                ranger(j, n-1, 0) {
                    ed[i][j] = min<int>(ed[i+1][j], ed[i][j+1]) + 1;
                    int cost = (s1[i] != s2[j]);
                    ed[i][j] = min<int>(ed[i][j], cost + ed[i+1][j+1]);
                }
            
            return ed[0][0];
        }
        
        static double kGramOverlap ( const string & S1, const string & S2 , int k = 2 ) {
            int m = S1.size(), n = S2.size();
            if ( m < k || n < k )   return 0.;

            auto s1 = tolower(S1), s2 = tolower(S2);
            unordered_set<string> g1, uniongrams, intersecgrams;

            range(i, 0, m-1) {
                int j = i + k - 1;
                if ( j >= m )   break;
                string gram = s1.substr(i, k);
                g1.insert(gram);
                uniongrams.insert(gram);
            }

            range(i, 0, n-1) {
                int j = i + k - 1;
                if ( j >= n )   break;
                string gram = s2.substr(i, k);
                uniongrams.insert(gram);
                if ( g1.find(gram) != g1.end() )
                    intersecgrams.insert(gram);
            }

            return 1. * intersecgrams.size() / uniongrams.size();
        }
        
        static double KLDivergence ( const string & s1, const string & s2 ) {
            vec<double> f1(26, 1.), f2(26, 1.);
            for ( char c : s1 ) {
                char l = tolower(c);
                int i = c - 'a';
                if ( i < 26 && i >= 0 )   f1[i]+=1.;
            }
            for ( char c : s2 ) {
                char l = tolower(c);
                int i = c - 'a';
                if ( i < 26 && i >= 0 )   f2[i]+=1.;
            }
            for ( auto & v : f1 )   v /= (26 + s1.size());
            for ( auto & v : f2 )   v /= (26 + s2.size());
            
            double kld = 0.;
            range(i, 0, 25) {
                double r = f1[i] / f2[i];
                kld += f1[i] * log10l(r);
            }
            return fabs(kld);
        }

        static double MaxSubstringOverlap ( const string & S1, const string & S2 ) {
            auto s1 = tolower(S1), s2 = tolower(S2);
            return 1. * LCS(s1, s2) / min<int>(s1.size(), s2.size());
        }

        static double FreqVecSimilarity ( const string & s1, const string & s2 ) {
            vec<int> f1(26, 0), f2(26, 0);
            for ( char c : s1 ) {
                char l = tolower(c);
                int i = c - 'a';
                if ( i < 26 && i >= 0 )   f1[i] ++;
            }
            for ( char c : s2 ) {
                char l = tolower(c);
                int i = c - 'a';
                if ( i < 26 && i >= 0 )   f2[i] ++;
            }
            double dotp = 0;
            range(i, 0, 25) 
                dotp += f1[i] * f2[i];
            dotp /= vecL2Norm(f1);
            dotp /= vecL2Norm(f2);
            return dotp;
        }

        static double ShortFormJaccardOverlap ( const string & s1, const string & s2 ) {
            return kGramOverlap(shortform(s1), shortform(s2), 1);
        }

        static int ShortFormEditDistance ( const string & s1, const string & s2 ) {
            return EditDistance(shortform(s1), shortform(s2));
        }

        static double SoundexSimilarity ( const string & s1, const string & s2 ) {
            return kGramOverlap(soundex(s1), soundex(s2), 2);
        }

        static double JaccardSimilarity ( const string & s1, const string & s2 ) {
            return kGramOverlap(s1, s2, 1);
        }
};


inline void activity ( const string & str ) {
    if ( verbose )  cout << str << endl;
}


class KnowledgeGraph {
    private:
        const string file_location;
        vec<triple_t> kg_triples;
        unordered_map<string, int> name2uri;
        vec<string> uri2name;
        unordered_map<int, vec<string>> pred_constraints;
        unordered_map<int, int> pred_types;
        unordered_set<int> entities;
        vec<vec<int>> entities_by_len;
        vec<vec<int>> entities_by_short_form_len;
        int next_anonymous_id;
        const int min_indexed_shortform_length = 3;

        int Add ( const string & s , bool isentity = 0 ) {
            if ( name2uri.find(s) != name2uri.end() )
                return name2uri.at(s);
            int id = name2uri.size();
            name2uri.insert({s, id});
            uri2name.push_back(s);
            if ( isentity ) AddEntities({s});
            return id;
        }

        string AddAnonymous ( ) {
            string name = "_:" + to_string(next_anonymous_id);
            next_anonymous_id ++;
            Add(name);
            return name;
        }

        static bool IsAnonymous ( const string & s ) {
            if(s.size()<2)  return 0;
            return (s[0] == '_' && s[1] == ':');
        }

        void print ( ) const {
            cout << " +++ { TRIPLES } +++\n";
            for ( auto & t : kg_triples ) {
                cout << "   <" << uri2name[t[0]] << "> " << uri2name[t[1]] << " ";
                int type = pred_types.at(t[1]);
                if ( type == STRING )  cout << '"' << uri2name[t[2]] << "\"\n";
                else if ( type == ENTITY )  cout << '<' << uri2name[t[2]] << ">\n";
                else    cout << uri2name[t[2]] << endl;
            }

            cout << "\n +++ { TYPES } +++\n";
            for ( auto & t : pred_types ) 
                cout << "   " << uri2name[t.first] << ' ' << TYPENAME[t.second] << endl;
            
            cout << "\n +++ { CONSTRAINTS } +++\n";
            for ( auto & c : pred_constraints )
                for ( auto & constr : c.second )
                    cout << "   " << uri2name[c.first] << ' ' << constr << endl;
        }

        void AddEntities ( const unordered_set<string> & l ) {
            int maxlen = 0;
            for ( auto & s : l )
                maxlen = max<int>(maxlen, s.size());
            
            if ( entities_by_len.size() - 1 < maxlen )
                entities_by_len.resize(maxlen+1);
            
            for ( auto & s : l ) {
                int si = name2uri.at(s);
                if ( entities.find(si) != entities.end() )   continue;

                string short_form = LinguisticMatching::shortform(s);
                int shortlen = short_form.size();
                
                if ( shortlen >= min_indexed_shortform_length ) {
                    if ( entities_by_short_form_len.size() < shortlen + 1 )
                        entities_by_short_form_len.resize(shortlen + 1);
                    entities_by_short_form_len[shortlen].push_back(si);
                }

                entities_by_len[s.size()].push_back(si);
                entities.insert(si);
            }
        }

        void read_triple ( fstream & kg , string & s , string & p , string & o ) {
            string line;
            getline(kg, line);

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
        }

        void removedup ( ) {
            set<triple_t> dist;
            for ( auto & t : kg_triples )
                dist.insert(t);
            kg_triples = vec<triple_t>(dist.begin(), dist.end());
        }

    public:
        friend class SchemaMapper;
        friend class RecordLinker;
        friend class Converter;

        KnowledgeGraph ( const string & filepath ) : file_location(filepath) {
            activity(" [ LOADING KNOWLEDGE GRAPH ]");
            
            kg_triples={}; name2uri={}; uri2name={}; 
            pred_constraints={}; pred_types={}; entities={};
            fstream kg; kg.open(filepath.c_str());
            unordered_set<string> anonymous_nodes;
            string s, p, o;

            read_triple(kg, s, p, o);
            while ( s != "-1" ) {
                int si = Add(s);
                int pi = Add(p);
                int oi = Add(o);
                kg_triples.push_back({si, pi, oi});
                if(IsAnonymous(s))  anonymous_nodes.insert(s);
                if(IsAnonymous(o))  anonymous_nodes.insert(o);
                read_triple(kg, s, p, o);
            }
            next_anonymous_id = anonymous_nodes.size();

            string constr_type;
            float constr_val;
            kg >> p;
            while ( p != "-1" ) {
                kg >> constr_type;
                int pi = Add(p);
                
                if ( pred_constraints.find(pi) == pred_constraints.end() )
                    pred_constraints.insert({pi, {}});
                auto & constr_vec = pred_constraints.at(pi);

                if ( constr_type == "unique" ) 
                    constr_vec.push_back(constr_type);
                else {
                    kg >> constr_val;
                    constr_vec.push_back(constr_type + " " + to_string(constr_val));
                }
                kg >> p;
            }

            for ( auto & v : pred_constraints )
                sort(v.second.begin(), v.second.end());

            string type;
            kg >> p;
            while ( p != "-1" ) {
                kg >> type;
                int pi = Add(p);
                pred_types.insert({pi, TYPEID.at(type)});
                kg >> p;
            }

            kg.close();

            int longest_len = 1;
            for ( auto & t : kg_triples ) {
                if ( pred_types[t[1]] == ENTITY ) {
                    longest_len = max<int>(longest_len, uri2name[t[2]].size());
                    if ( ! IsAnonymous(uri2name[t[2]]) )
                        entities.insert(t[2]);
                }
                longest_len = max<int>(longest_len, uri2name[t[0]].size());
                if ( ! IsAnonymous(uri2name[t[0]]) )
                    entities.insert(t[0]);
            }

            entities_by_len = vec<vec<int>>(longest_len+1);
            for ( int uri : entities ) 
                entities_by_len[uri2name[uri].size()].push_back(uri);
        }

        void save ( ) {
            removedup();
            fstream kg; kg.open(file_location.c_str());

            for ( auto & t : kg_triples ) {
                kg << "<" << uri2name[t[0]] << "> ";
                kg << uri2name[t[1]] << " ";

                int type = pred_types.at(t[1]);
                if ( type == STRING )       kg << '"' << uri2name[t[2]] << '"';
                else if ( type == ENTITY )  kg << '<' << uri2name[t[2]] << '>';
                else                        kg << uri2name[t[2]];
                
                kg << endl;
            }
            kg << "-1\n";

            for ( auto & c : pred_constraints ) {
                auto & pred = uri2name[c.first];
                for ( auto & s : c.second )
                    kg << pred << ' ' << s << endl;
            }
            kg << "-1\n";

            for ( auto & pt : pred_types ) {
                auto & pred = uri2name[pt.first];
                kg << pred << ' ' << TYPENAME[pt.second] << endl;
            }
            kg << "-1\n";

            kg.close();
        }
};


class Table {
    private:
        string table_name;
        unordered_set<string> primary_keys;
        unordered_map<string, int> col_types;
        unordered_map<string, vec<string>> col_constraints;
        vec<string> columns;
        vec<row_t> rows;

        void AddConstraint ( const string & type , const string & col , fstream & dbstream ) {
            int coltype = col_types.at(col);

            if(type == "primary")
                primary_keys.insert(col);
            
            else if ( type == "unique" ) {
                if(col_constraints.find(col) == col_constraints.end())
                    col_constraints.insert({col, {"unique"}});
                else
                    col_constraints.at(col).push_back("unique");
            }
            
            else if ( coltype == FLOAT || coltype == INT ) {
                float val; dbstream >> val;
                string c = type + " " + to_string(val);
                if(col_constraints.find(col) == col_constraints.end())
                    col_constraints.insert({col, {c}});
                else
                    col_constraints.at(col).push_back(c);
            }

            else {
                string val; dbstream >> val;
                string c = type + " " + val;
                if(col_constraints.find(col) == col_constraints.end())
                    col_constraints.insert({col, {c}});
                else
                    col_constraints.at(col).push_back(c);
            }
        }

        void parse_row ( fstream & dbstream ) {
            string row_string;
            row_t row;
            getline(dbstream, row_string);

            int i = 0;
            int len = row_string.size();

            while ( i < len ) {
                string cell;
                if ( row_string[i] == '<' ) {
                    i ++;
                    while ( row_string[i] != '>' )
                        cell.push_back(row_string[i++]);
                    i += 2;
                }

                else if ( row_string[i] == '"' ) {
                    i ++;
                    while ( row_string[i] != '"' )
                        cell.push_back(row_string[i++]);
                    i += 2;
                }

                else {
                    while ( i < len && row_string[i] != ' ' )
                        cell.push_back(row_string[i++]);
                    i ++;
                }

                row.push_back(cell);
            }

            rows.push_back(row);
        }

        void print ( ) const {
            cout << "\n +++ { ROWS } +++\n";
            cout << "   " << columns[0];
            int numcols = columns.size();
            range(i, 1, numcols-1)
                cout << ", " << columns[i];
            cout << endl;

            for ( auto & row : rows ) {
                cout << "   " << row[0];
                range(i, 1, numcols-1)
                    cout << ", " << row[i];
                cout << endl;
            }

            cout << "\n +++ { TYPES } +++\n";
            for ( auto & t : col_types )
                cout << "   " << t.first << " - " << TYPENAME[t.second] << endl;

            cout << "\n +++ { CONSTRAINTS } +++\n";
            for ( auto & col : primary_keys )
                cout << "   " << col << " - primary\n";
            for ( auto & t : col_constraints ) {
                for ( auto & c : t.second )
                    cout << "   " << t.first << " - " << c << endl;
            }
        }

    public:
        friend class SchemaMapper;
        friend class RecordLinker;
        friend class DataBase;
        friend class Converter;

        Table ( ) { col_types = {{"type", ENTITY}}; }

        void fill ( fstream & dbstream ) {
            string bin;
            dbstream >> table_name;

            activity(" [ LOADING TABLE " + table_name + " ]");

            int num_cols; dbstream >> num_cols;
            columns = vec<string>(num_cols);
            string colname, type;

            while ( num_cols -- ) {
                string colname, type;
                dbstream >> colname >> type;
                col_types.insert({colname, TYPEID.at(type)});
            }

            int num_constrs; dbstream >> num_constrs;
            string constr_type, col;
            while ( num_constrs -- ) {
                dbstream >> constr_type >> col;
                AddConstraint(constr_type, col, dbstream);
            }

            int num_rows; dbstream >> num_rows;
            for ( auto & col : columns )    dbstream >> col;
            getline(dbstream, bin);
            while ( num_rows -- )   parse_row(dbstream);

            if ( verbose )  print();
        }
};


class SchemaMapper {
    private:
        map<string, int> mapping;
        const Table & table;
        KnowledgeGraph & kgr;
        const static int thresh = 7;

        static int DataTypeClass ( int t ) {
            if ( t == INT || t == FLOAT )   return -1;
            return t;
        }

        static bool MatchingTypes ( int t1 , int t2 ) {
            return (DataTypeClass(t1) == DataTypeClass(t2));
        }

        bool MatchingConstraints ( const string & col , int pred ) {
            if ( table.col_constraints.find(col) == table.col_constraints.end() ) {
                if ( kgr.pred_constraints.find(pred) == kgr.pred_constraints.end() )    return 1;
                return 0;
            }
            if ( kgr.pred_constraints.find(pred) == kgr.pred_constraints.end() )    return 0;
            return (table.col_constraints.at(col) == kgr.pred_constraints.at(pred));
        }

        bool MatchingLinguistics ( const string & col , int pred ) {
            auto predname = kgr.uri2name[pred];
            return (LinguisticMatching::EditDistance(col, predname) <= thresh);
        }

        bool SuggestSchemaMapping ( const string & col , int pred ) const {
            cout << "   { Schema Mapping }  Map `" << col << "` to `" << kgr.uri2name[pred] << "` ? (Y/N) ";
            char ans;   cin >> ans;
            return (ans == 'Y');
        }

        void Handle ( const string & col ) {
            if ( kgr.name2uri.find(col) != kgr.name2uri.end() ) {
                int uri = kgr.name2uri.at(col);
                if ( kgr.pred_types.find(uri) != kgr.pred_types.end() ) {
                    mapping.insert({col, uri});
                    return;
                }
            }
            
            if ( table.primary_keys.find(col) != table.primary_keys.end() &&
                 table.primary_keys.size() == 1 )   return;

            int type = table.col_types.at(col);
            for ( auto & pred_type : kgr.pred_types ) {
                if ( kgr.uri2name[pred_type.first] == "type" )  continue;
                
                if(MatchingTypes(type, pred_type.second) && MatchingConstraints(col, pred_type.first) && 
                   MatchingLinguistics(col, pred_type.first)) {
                    bool mapped = SuggestSchemaMapping(col, pred_type.first);
                    if ( mapped ) {
                        mapping.insert({col, pred_type.first});
                        return;
                    }
                }
            }
        }

    public:
        friend class Converter;

        SchemaMapper ( const Table & t , KnowledgeGraph & kg ) : table(t), kgr(kg) { }
        
        void run ( ) { for ( auto & col : table.columns ) Handle(col); }
};


class DecisionTree {
    private:
        DecisionTree * left, * right;
        bool result;
        int attribute;
        double threshold;

        static vec<double> CandidateThresholds ( const vec<pair<vec<double>, bool>> & data , int a ) {
            set<double> vals;
            for ( auto & x : data )
                vals.insert(x.first[a]);

            int l = vals.size();
            if ( l == 1 ) return {};
            
            vec<double> valvec (vals.begin(), vals.end());
            vec<double> threshs;    threshs.reserve(l-1);
            range(i, 0, l-2) {
                double t = (valvec[i] + valvec[i+1]) / 2;
                threshs.push_back(t);
            }
            return threshs;
        }

        static double entropy ( int yes , int no ) {
            if ( yes == 0 || no == 0 )  return 0;
            int n = yes + no;
            double r1 = 1. * yes / n;
            double r2 = 1. - r1;
            return -1 * (r1 * log2l(r1) + r2 * log2l(r2));
        }

        void SplitDecisionTree ( const vec<pair<vec<double>, bool>> & data ) {
            double least_entropy = 1.e5;
            int flen = data[0].first.size();

            range(a, 0, flen-1) {
                auto threshs = CandidateThresholds(data, a);
                
                for ( double t : threshs ) {
                    int y1 = 0, n1 = 0, y2 = 0, n2 = 0;
                
                    for ( auto & x : data ) {
                        if ( x.first[a] <= t ) {
                            if ( x.second ) y1 ++;
                            else    n1 ++;
                        }
                        else {
                            if ( x.second ) y2 ++;
                            else    n2 ++;
                        }
                    }

                    double e1 = entropy(y1, n1);
                    double e2 = entropy(y2, n2);
                    double r = 1. * (y1 + n1) / data.size();
                    double e = r * e1 + (1 - r) * e2;

                    if ( e < least_entropy ) {
                        least_entropy = e;
                        attribute = a; 
                        threshold = t;
                    }
                }
            }

            vec<pair<vec<double>, bool>> data1, data2;
            for ( auto & x : data ) {
                if ( x.first[attribute] <= threshold )  data1.push_back(x);
                else    data2.push_back(x);
            }

            left = new DecisionTree;    left->fit(data1);
            right = new DecisionTree;   right->fit(data2);
        }

    public:
        class RandomForest;

        DecisionTree ( ) : left(0), right(0) {}

        void fit ( const vec<pair<vec<double>, bool>> & data ) {
            delete left;
            delete right;
            int yes = 0;
            for ( auto & x : data ) yes += x.second;
            if ( yes == 0 ) { result = 0; left = right = NULL; }
            else if ( yes == data.size() )
                { result = 1; left = right = NULL; }
            else    SplitDecisionTree(data);
        }

        ~DecisionTree ( ) {
            delete left;
            delete right;
        }

        bool Result ( const vec<double> & sample ) const {
            if ( ! left )   return result;
            if ( sample[attribute] <= threshold )
                return left->Result(sample);
            return right->Result(sample);
        }
};


class RandomForest {
    private:
        vec<DecisionTree> classifiers;

        static vec<pair<vec<double>, bool>> Bagging ( vec<pair<vec<double>, bool>> data , int i , int j ) {
            int n = j - i + 1;
            vec<pair<vec<double>, bool>> bag(n);
            range(k, 0, n-1)
                bag[k] = data[Random(0,n-1)+i];
            return bag;
        }

    public:
        const static int forestsize;
        friend class RandomForestActiveLearner;
        friend class RecordLinker;

        RandomForest ( ) : classifiers({}) { }

        void fit ( const vec<pair<vec<double>, bool>> & data , int i , int j ) {
            classifiers = vec<DecisionTree>(forestsize);
            for ( auto & c : classifiers )  c.fit(Bagging(data, i, j));
        }

        bool Result ( const vec<double> & sample ) const {
            int yes = 0, no = 0;
            for ( auto & d : classifiers )
                yes += d.Result(sample);
            no = forestsize - yes;
            return (yes > no);
        }

        int NumVotes ( const vec<double> & sample ) const {
            int yes = 0, no = 0;
            for ( auto & d : classifiers )
                yes += d.Result(sample);
            no = forestsize - yes;
            return max<int>(yes, no);
        }

        double ConfidenceScore ( const vec<double> & sample ) const {
            return 1. * NumVotes(sample) / forestsize;
        }

        double Accuracy ( const vec<pair<vec<double>, bool>> & data , int i , int j ) const {
            int correct = 0;
            range(k, i, j) {
                auto & x = data[k];
                if ( Result(x.first) == x.second )  correct ++;
            }
            return 1. * correct / data.size();
        }
};

const int RandomForest::forestsize = 7;

class RandomForestActiveLearner {
    private:
        vec<pair<string, string>> labelledlinkings;
        map<vec<double>, bool> labelled_map;
        vec<pair<vec<double>, bool>> labelled;
        set<pair<double, rfal_sample_t>> unlabelled;
        RandomForest classifier;
        const int labelledInc = 2;
        const int densityKVal = 5;
        const static int initial = 5;
        const static int maxiters = 5;
        constexpr static double termaccuracy = 0.70;
        constexpr static double trainpartition = 0.65;
        constexpr static double smoothing = 0.05;

        static void shuffle ( vec<int> & v , int i = 0 ) {
            if ( i == v.size() - 1 )    return;
            int j = Random(i, v.size()-1);
            swap(v[i], v[j]);
            shuffle(v, i+1);
        }

        void shuffle_labelled ( int i = 0 ) {
            if ( i == labelled.size() - 1 )    return;
            int j = Random(i, labelled.size()-1);
            swap(labelled[i], labelled[j]);
            shuffle_labelled(i+1);
        }

        void Initialize ( ) {
            auto it = unlabelled.begin();
            advance(it, Random(0, unlabelled.size()-1));

            auto & feature = featVec(it->second);
            
            bool label;
            if ( labelled_map.find(feature) != labelled_map.end() )
                label = labelled_map.at(feature);
            else {
                label = LabelManually(firstEnt(it->second), secEnt(it->second));
                labelled_map.insert({feature, label});
            }

            labelled.push_back({feature, label});
            if ( label )
                labelledlinkings.push_back({firstEnt(it->second), secEnt(it->second)});
            
            unlabelled.erase(it);

            LabelNextK(initial-1);
        }

        int uncertainity ( const vec<double> & sample ) const {
            return 2 * classifier.NumVotes(sample) - RandomForest::forestsize;
        }

        static double distance ( const vec<double> & s1 , const vec<double> & s2 ) {
            int l = s1.size();
            double d = 0;
            range(i, 0, l-1) {
                double g = s1[i] - s2[i];
                d += g * g;
            }
            return d;
        }

        double density ( const vec<double> & sample ) const {
            int n = unlabelled.size();
            priority_queue<double> dists;
            for ( auto & x : unlabelled ) {
                double d = this->distance(sample, featVec(x.second));
                if ( dists.size() < densityKVal )   dists.push(d);
                else if ( dists.top() > d ) {
                    dists.pop();
                    dists.push(d);
                }
            }
            double den = 0;
            int m = dists.size();
            while ( ! dists.empty() ) {
                den += dists.top();
                dists.pop();
            }
            return den / m;
        }

        double diversity ( const vec<double> & sample ) const {
            double div = 1e10;
            for ( auto & x : labelled )
                div = min<double>(div, distance(sample, x.first));
            return div;
        }

        double SelectionMetric ( const vec<double> & x ) const {
            return uncertainity(x) + density(x) - diversity(x);
        }

        void LabelNextK ( int k ) {
            if ( k <= 0 )   return;

            priority_queue<pair<double, pair<double, rfal_sample_t>>> topk;
            for ( auto & score_sample : unlabelled ) {
                double score = score_sample.first;
                auto & sample = score_sample.second;
                
                double v = SelectionMetric(featVec(sample)) + smoothing * score;
                if ( topk.size() < k )  topk.push({v, score_sample});
                
                else if ( v < topk.top().first ) {
                    topk.pop();
                    topk.push({v, score_sample});
                }
            }
            
            while ( ! topk.empty() ) {
                auto & chosen = topk.top().second;
                unlabelled.erase(chosen);

                auto & feature = featVec(chosen.second);
                auto & e1 = firstEnt(chosen.second);
                auto & e2 = secEnt(chosen.second);

                bool label;
                if ( labelled_map.find(feature) != labelled_map.end() ) 
                    label = labelled_map.at(feature);
                else {
                    label = LabelManually(e1, e2);
                    labelled_map.insert({feature, label});
                }
                
                labelled.push_back({featVec(chosen.second), label});

                if ( label )
                    labelledlinkings.push_back({e1, e2});

                topk.pop();
            }
        }

        bool ActiveLearning ( ) {
            if ( unlabelled.size() == 0 )   return 0;

            shuffle_labelled();
            int len = labelled.size();
            int train_len = floor(trainpartition * len);
            
            classifier.fit(labelled, 0, train_len-1);
            double acc = classifier.Accuracy(labelled, train_len, len-1);

            activity("  [ ACCURACY : " + to_string(acc * 100) + " % ]");
            return ( acc < termaccuracy );
        }

    public:
        friend class RecordLinker;

        RandomForestActiveLearner ( ) { }

        void learn ( const set<pair<double, rfal_sample_t>> & samples ) {
            unlabelled = samples;
            
            activity("  [ INITIALIZATION ]");
            Initialize();
            
            bool proceed = 1;
            int iter = 0;

            while ( iter <= maxiters ) {
                activity("\n  [ ITERATION " + to_string(iter+1) + " ]");
                activity("  [ # OF UNLABELLED " + to_string(unlabelled.size()) + " ]");
                proceed = ActiveLearning();
                iter ++;
                if ( iter == maxiters || ! proceed )    break;
                LabelNextK(min<int>(unlabelled.size(), labelledInc));
            }
        }

        static bool LabelManually ( const string & e1 , const string & e2 ) {
            cout << "   { Active Learning }  Link <" << e1 << "> to <" << e2 << "> ? (Y/N) ";
            char ans;   cin >> ans;
            return (ans == 'Y');
        }
};


class RecordLinker {
    private:
        unordered_map<string, string> linking;
        const Table & table;
        const KnowledgeGraph & kgr;
        unordered_set<string> table_entities;
        constexpr static double fullform_editdist_ratio = 0.25;
        constexpr static double shortform_editdist_ratio = 0.40;
        
        void CollectTableEntities ( ) {
            table_entities = {};
            int i = -1;
            for ( auto & col : table.columns ) {
                i ++;
                if ( table.col_types.at(col) != ENTITY )    continue;
                for ( auto & row : table.rows )
                    table_entities.insert(row[i]);
            }
        }

        static vec<double> PairFeatureVector ( const string & e1 , const string & e2 ) {
            int ed = LinguisticMatching::EditDistance(e1, e2);
            return {
                1. * ed / min<int>(e1.size(), e2.size()),
                LinguisticMatching::JaccardSimilarity(e1, e2),
                LinguisticMatching::kGramOverlap(e1, e2),
                LinguisticMatching::MaxSubstringOverlap(e1, e2),
                LinguisticMatching::FreqVecSimilarity(e1, e2),
                LinguisticMatching::KLDivergence(e1, e2)
            };
        }

        void GetLinkings ( const RandomForestActiveLearner & active_learner ) {
            unordered_map<string, int> maxvotes;

            for ( auto & p : active_learner.labelledlinkings ) {
                linking.insert({p.first, p.second});
                maxvotes.insert({p.first, RandomForest::forestsize});
            }

            for ( auto & x : active_learner.unlabelled ) {
                auto & e1 = firstEnt(x.second);
                auto & e2 = secEnt(x.second);
                
                bool label = active_learner.classifier.Result(featVec(x.second));
                if ( ! label )  continue;
                
                int votes = active_learner.classifier.NumVotes(featVec(x.second));
                if ( maxvotes.find(e1) == maxvotes.end() ) {
                    maxvotes.insert({e1, votes});
                    linking.insert({e1, e2});
                }
                else if ( votes > maxvotes.at(e1) ) {
                    maxvotes.at(e1) = votes;
                    linking.at(e1) = e2;
                }
            }

            activity("\n [ RESULTS ]");

            if ( maxvotes.size() == 0 ) {
                activity("  NIL");
                return;
            }
            
            string fs = to_string(RandomForest::forestsize);
            for ( auto & x : maxvotes )
                activity("  Linked <" + x.first + "> to <" + linking.at(x.first) + 
                         "> (confidence : " + to_string(x.second) + "/" + fs +  ")");
        }

        void FullFormCandidates ( const string & e , unordered_map<int, double> & ss , bool isshort = 0 ) const {
            int len = e.size();
            double edit_thresh_ratio = isshort ? shortform_editdist_ratio : fullform_editdist_ratio;
            int edit_thresh = ceil(edit_thresh_ratio * len);
            int len_low = max<int>(len - edit_thresh, 1);
            int len_high = min<int>(len + edit_thresh, kgr.entities_by_len.size()-1);

            range(l, len_low, len_high) 
                for ( int uri : kgr.entities_by_len[l] ) {
                    const string & e2 = kgr.uri2name[uri];
                    double rat = 1. * LinguisticMatching::EditDistance(e, e2) / min<int>(len, e2.size());
                    if ( rat <= edit_thresh_ratio ) ss.insert({uri, rat});
                }
        }

        void ShortFormCandidates ( const string & e , unordered_map<int, double> & ss ) const {
            int len = e.size();
            int edit_thresh = ceil(shortform_editdist_ratio * len);
            int len_low = max<int>(len - edit_thresh, 1);
            int len_high = min<int>(len + edit_thresh, kgr.entities_by_short_form_len.size()-1);

            range(l, len_low, len_high) 
                for ( int uri : kgr.entities_by_short_form_len[l] ) {
                    const string & e2 = LinguisticMatching::shortform(kgr.uri2name[uri]);
                    double rat = 1. * LinguisticMatching::EditDistance(e, e2) / min<int>(len, e2.size());
                    if ( rat <= shortform_editdist_ratio )   ss.insert({uri, rat});
                }
            
            string shortform = LinguisticMatching::shortform(e);
            FullFormCandidates(shortform, ss, 1);
        }

        unordered_map<int, double> GetCandidatePairs ( const string & e ) const {
            unordered_map<int, double> candidates;
            FullFormCandidates(e, candidates);
            ShortFormCandidates(e, candidates);
            return candidates;
        }

    public:
        friend class Converter;

        RecordLinker ( const Table & t , const KnowledgeGraph & kg ) 
            : table(t), kgr(kg), linking({}) { CollectTableEntities(); }

        void run ( ) {
            set<pair<double, rfal_sample_t>> samples;

            for ( auto & e1 : table_entities ) {
                if ( kgr.name2uri.find(e1) != kgr.name2uri.end() &&
                     kgr.entities.find(kgr.name2uri.at(e1)) != kgr.entities.end() )
                    continue;
                
                const auto & cands = GetCandidatePairs(e1);
                for ( auto & uri_score : cands ) {
                    int uri = uri_score.first;
                    double score = uri_score.second;
                    auto & e2 = kgr.uri2name[uri];
                    rfal_sample_t feature = {e1, e2, PairFeatureVector(e1, e2)};
                    samples.insert({score, feature});
                }
            }

            activity(" [ # OF SAMPLES : " + to_string(samples.size()) + " ]\n");

            if ( samples.size() == 0 )  return;

            
            RandomForestActiveLearner active_learner;

            if ( samples.size() <= RandomForestActiveLearner::initial ) {
                for ( auto & x : samples ) {
                    auto & e1 = firstEnt(x.second);
                    auto & e2 = secEnt(x.second);
                    bool label = RandomForestActiveLearner::LabelManually(e1, e2);
                    if ( label )    active_learner.labelledlinkings.push_back({e1, e2});
                }
            }
            else 
                active_learner.learn(samples);
            
            GetLinkings(active_learner);
        }
};


class DataBase {
    private:
        int num_tables;
        vec<Table> tables;

    public:
        friend class Converter;

        DataBase ( ) : tables({}) { }
        
        void fill ( const string & file_location ) {
            activity(" [ LOADING DATABASE ]");
            fstream dbstream; 
            dbstream.open(file_location.c_str());
            
            dbstream >> num_tables;
            tables = vec<Table>(num_tables);
            for ( Table & table : tables ) {
                activity(string(40, '-'));
                table.fill(dbstream);
                activity(string(40, '-') + "\n");
            }
        }
};


class Converter {
    private:
        const DataBase & db;
        KnowledgeGraph & kgr;

        vec<vec<string>> getTriples ( const Table & table ) {
            const auto & cols = table.columns;
            const auto & rows = table.rows;
            int n = cols.size();
            
            bool onePrimKey = (table.primary_keys.size() == 1);
            string key = *table.primary_keys.begin();
            
            int keyidx = -1;
            if ( onePrimKey ) {
                keyidx = 0;
                while ( cols[keyidx] != key )   keyidx ++;
            }

            vec<vec<string>> triples;
            for ( auto & row : rows ) {
                string subject;
                
                if ( onePrimKey ) {
                    subject = row[keyidx];
                    triples.push_back({subject, "type", key});
                }
                else
                    subject = kgr.AddAnonymous();
                
                range(c, 0, n-1) {
                    if ( c == keyidx )  continue;
                    triples.push_back({subject, cols[c], row[c]});
                }
            }

            return triples;
        }

        void integrate ( const SchemaMapper & sm , RecordLinker & rl ) {
            auto & table = sm.table;
            auto triples = getTriples(table);

            for ( auto & triple : triples ) {
                auto & pred = triple[1];
                int pred_id;

                if ( sm.mapping.find(pred) != sm.mapping.end() ) {
                    int mapped_pred_id = sm.mapping.at(pred);
                    auto & mapped_pred_type = kgr.pred_types.at(mapped_pred_id);
                    mapped_pred_type = min<int>(mapped_pred_type, table.col_types.at(pred));
                    pred_id = mapped_pred_id;
                    pred = kgr.uri2name[mapped_pred_id];
                }
                else {
                    pred_id = kgr.Add(pred);
                    kgr.pred_types.insert({pred_id, table.col_types.at(pred)});
                    if ( table.col_constraints.find(pred) != table.col_constraints.end() ) {
                        auto & constraints = table.col_constraints.at(pred);
                        kgr.pred_constraints.insert({pred_id, constraints});
                    }
                }
                
                auto & subj = triple[0];
                auto & obj = triple[2];
                int pred_type = kgr.pred_types.at(pred_id);

                if ( rl.linking.find(subj) != rl.linking.end() ) 
                    subj = rl.linking.at(subj);
                else    kgr.Add(subj, 1);
                
                if ( rl.linking.find(obj) != rl.linking.end() ) 
                    obj = rl.linking.at(obj);
                else if ( pred_type == ENTITY ) kgr.Add(obj, 1);
                else    kgr.Add(obj, 0);
            }

            for ( auto & triple : triples ) {
                int si = kgr.name2uri.at(triple[0]);
                int pi = kgr.name2uri.at(triple[1]);
                int oi = kgr.name2uri.at(triple[2]);
                kgr.kg_triples.push_back({si, pi, oi});
            }

            auto & entities = rl.table_entities;
            for ( auto & p : rl.linking )
                entities.erase(p.first);

            kgr.AddEntities(entities);
        }
    
    public:
        Converter ( const DataBase & d , KnowledgeGraph & kg ) : db(d), kgr(kg) { }

        void run ( ) {
            for ( auto & table : db.tables ) {
                SchemaMapper sm(table, kgr);
                RecordLinker rl(table, kgr);
                
                activity(string(40, '-'));
                activity(" [ MAPPING SCHEMA FOR TABLE " + table.table_name + " ]\n");
                sm.run(); 
                activity("\n [ SCHEMA MAPPING FINISHED ]");
                activity(string(40, '-') + "\n");
                
                activity(string(40, '-'));
                activity(" [ LINKING RECORDS FOR TABLE " + table.table_name + " ]");
                rl.run();
                activity("\n [ RECORD LINKING FINISHED ]");
                activity(string(40, '-') + "\n");
                
                integrate(sm, rl);
                activity(" [ KNOWLEDGE GRAPH UPDATED ]");
                activity(string(40, '-') + "\n");
            }
        }
};


int main ( int argc , char ** argv ) 
{
    cout << "\n\n";
    KnowledgeGraph kg(argv[1]);
    
    DataBase db;
    db.fill(argv[2]);

    Converter converter(db, kg);
    converter.run();

    kg.save();
    cout << "\n\n";
}
