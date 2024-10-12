<?php

namespace App\Http\Controllers;

use App\Models\Suspicious;
use App\Models\Subscription;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class BarringController extends Controller
{
    public function suspicious()
    {
        $suspiciouses = Suspicious::whereNull("response")->orderBy('created_at', 'desc')->get();
        return view("webmaster.frauds.suspicious")->with('suspiciouses', $suspiciouses);
    }
    public function barred()
    {
        $barreds = Suspicious::where("response", "barred")->orderBy('created_at', 'desc')->get();
        return view("webmaster.frauds.barred")->with('barreds', $barreds);
    }
    public function fraud(Suspicious $suspicious)
    {
        Subscription::find($suspicious->subscription)->update([
            'barred_by' => Auth::user()->id,
            'barred_at' => now()
        ]);
        $suspicious->update([
            'checked_by' => Auth::user()->id,
            'response' => "barred",
            'checked_at' => now()
        ]);
        return back()->with("success", "Subscription has been barred successfully");
    }
    public function notfraud(Suspicious $suspicious)
    {
        $suspicious->update([
            'checked_by' => Auth::user()->id,
            'response' => "removed",
            'checked_at' => now()
        ]);
        return back()->with("success", "Subscription has been removed from the list");
    }
    public function reactive(Suspicious $barred)
    {
        Subscription::find($barred->subscription)->update([
            'barred_by' => null,
            'barred_at' => null
        ]);
        $barred->update([
            'response' => "reactivated",
            'checked_at' => now()
        ]);
        return back()->with("success", "Subscription has been reactived successfully");

    }
}
