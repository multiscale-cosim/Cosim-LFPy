/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ExpSynI
#define _nrn_initial _nrn_initial__ExpSynI
#define nrn_cur _nrn_cur__ExpSynI
#define _nrn_current _nrn_current__ExpSynI
#define nrn_jacob _nrn_jacob__ExpSynI
#define nrn_state _nrn_state__ExpSynI
#define _net_receive _net_receive__ExpSynI 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define tau _p[0]
#define tau_columnindex 0
#define i _p[1]
#define i_columnindex 1
#define q _p[2]
#define q_columnindex 2
#define onset_times (_p + 3)
#define onset_times_columnindex 3
#define weight_list (_p + 1003)
#define weight_list_columnindex 1003
#define v _p[2003]
#define v_columnindex 2003
#define _g _p[2004]
#define _g_columnindex 2004
#define _tsav _p[2005]
#define _tsav_columnindex 2005
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_exp_current(void*);
 static double _hoc_update_queue(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "exp_current", _hoc_exp_current,
 "update_queue", _hoc_update_queue,
 0, 0
};
#define exp_current exp_current_ExpSynI
#define update_queue update_queue_ExpSynI
 extern double exp_current( _threadargsprotocomma_ double );
 extern double update_queue( _threadargsprotocomma_ double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "tau", 1e-09, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau", "ms",
 "i", "nA",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ExpSynI",
 "tau",
 0,
 "i",
 "q",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 2006, _prop);
 	/*initialize range parameters*/
 	tau = 5;
  }
 	_prop->param = _p;
 	_prop->param_size = 2006;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _expsyni_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 2006, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ExpSynI /home/vagrant/multiscale-cosim/my_forks/Cosim-LFPy/science/mod/expsyni.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Exponential-function synaptic current, with NET_RECEIVE";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
double update_queue ( _threadargsprotocomma_ double _ln ) {
   double _lupdate_queue;
 double _lk ;
 {int  _lk ;for ( _lk = 0 ; _lk <= ((int) q ) - ((int) _ln ) - 1 ; _lk ++ ) {
     onset_times [ _lk ] = onset_times [ _lk + ((int) _ln ) ] ;
     weight_list [ _lk ] = weight_list [ _lk + ((int) _ln ) ] ;
     } }
   q = q - _ln ;
   
return _lupdate_queue;
 }
 
static double _hoc_update_queue(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  update_queue ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
double exp_current ( _threadargsprotocomma_ double _lx ) {
   double _lexp_current;
 if ( _lx < 0.0 ) {
     _lexp_current = 0.0 ;
     }
   else {
     _lexp_current = exp ( - _lx ) ;
     }
   
return _lexp_current;
 }
 
static double _hoc_exp_current(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  exp_current ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   onset_times [ ((int) q ) ] = t ;
   weight_list [ ((int) q ) ] = _args[0] ;
   if ( q >= 1000.0 - 1.0 ) {
     printf ( "Error in ExpSynI. Spike queue is full\n" ) ;
     }
   else {
     q = q + 1.0 ;
     }
   } }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   i = 0.0 ;
   q = 0.0 ;
   }

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   double _lk , _lexpired_spikes , _lx ;
 i = 0.0 ;
   _lexpired_spikes = 0.0 ;
   {int  _lk ;for ( _lk = 0 ; _lk <= ((int) q ) - 1 ; _lk ++ ) {
     _lx = ( t - onset_times [ _lk ] ) / tau ;
     if ( _lx > 20.0 ) {
       _lexpired_spikes = _lexpired_spikes + 1.0 ;
       }
     else {
       i = i - weight_list [ _lk ] * exp_current ( _threadargscomma_ _lx ) ;
       }
     } }
   update_queue ( _threadargscomma_ _lexpired_spikes ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/vagrant/multiscale-cosim/my_forks/Cosim-LFPy/science/mod/expsyni.mod";
static const char* nmodl_file_text = 
  "TITLE Exponential-function synaptic current, with NET_RECEIVE\n"
  "\n"
  "COMMENT\n"
  "This model works with variable time-step methods (although it may not\n"
  "be very accurate) but at the expense of having to maintain the queues\n"
  "of spike times and weights.\n"
  "Andrew P. Davison, UNIC, CNRS, May 2006\n"
  "\n"
  "Note: converted to Exponential current kernel. \n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "DEFINE MAX_SPIKES 1000\n"
  "DEFINE CUTOFF 20\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS ExpSynI\n"
  "	RANGE tau, i, q\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tau = 5 (ms) <1e-9,1e9>\n"
  "\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	i (nA)\n"
  "	q\n"
  "	onset_times[MAX_SPIKES] (ms)\n"
  "	weight_list[MAX_SPIKES] (nA)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	i  = 0\n"
  "	q  = 0 : queue index\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	LOCAL k, expired_spikes, x\n"
  "	i = 0\n"
  "	expired_spikes = 0\n"
  "	FROM k=0 TO q-1 {\n"
  "		x = (t - onset_times[k])/tau\n"
  "		if (x > CUTOFF) {\n"
  "			expired_spikes = expired_spikes + 1\n"
  "		} else {\n"
  "			i = i - weight_list[k] * exp_current(x)\n"
  "		}\n"
  "	}\n"
  "	update_queue(expired_spikes)\n"
  "}\n"
  "\n"
  "FUNCTION update_queue(n) {\n"
  "	LOCAL k\n"
  "	:if (n > 0) { printf(\"Queue changed. t = %4.2f onset_times=[\",t) }\n"
  "	FROM k=0 TO q-n-1 {\n"
  "		onset_times[k] = onset_times[k+n]\n"
  "		weight_list[k] = weight_list[k+n]\n"
  "		:if (n > 0) { printf(\"%4.2f \",onset_times[k]) }\n"
  "	}\n"
  "	:if (n > 0) { printf(\"]\\n\") }\n"
  "	q = q-n\n"
  "}\n"
  "\n"
  "FUNCTION exp_current(x) {\n"
  "	if (x < 0) {\n"
  "		exp_current = 0\n"
  "	} else {\n"
  "		exp_current = exp(-x)\n"
  "	}\n"
  "}\n"
  "\n"
  "NET_RECEIVE(weight (nA)) {\n"
  "	onset_times[q] = t\n"
  "	weight_list[q] = weight\n"
  "	if (q >= MAX_SPIKES-1) {\n"
  "		printf(\"Error in ExpSynI. Spike queue is full\\n\")\n"
  "	} else {\n"
  "		q = q + 1\n"
  "	}\n"
  "}\n"
  ;
#endif
