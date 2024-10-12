<?php

namespace App\Http\Controllers;

use webmaster;
use App\Models\Subscription;
use Illuminate\Http\Request;

class SubscriptionController extends Controller
{
    public function index()
    {
        $subscriptions = Subscription::orderBy('id', 'desc')->get();
        return view('webmaster.subscription.subscription')->with('subscriptions', $subscriptions);
    }
}
