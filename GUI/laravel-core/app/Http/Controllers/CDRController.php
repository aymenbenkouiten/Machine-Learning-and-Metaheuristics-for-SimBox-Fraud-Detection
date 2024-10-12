<?php

namespace App\Http\Controllers;

use App\Models\CDR;
use Illuminate\Http\Request;

class CDRController extends Controller
{
    public function index()
    {
        $records = CDR::orderBy('id', 'desc')->limit(500)->get();
        return view('webmaster.cdr.cdr')->with('records', $records);
    }
}
