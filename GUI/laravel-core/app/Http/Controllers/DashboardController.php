<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Models\Subscription;
use Illuminate\Http\Request;

class DashboardController extends Controller
{
    public function index()
    {
        $subscriptions = Subscription::inRandomOrder()->take(10)->get();
        $sortedSubscriptions = $subscriptions->sortByDesc(function ($subscription) {
            $lastThreeDigits = substr($subscription->phone_number, -3);
            return (int)$lastThreeDigits;
        });

        $users = User::all();
        $logs = [];
        $startDate = today();
        $endDate = today()->addDays(2);
        for($i=0;$i<10;$i++)
        {
            $log['user'] = $users->random()->name;
            $log['section'] = ["Dashboard", "Customers", "Subscribtions", "Call details recores", "Suspicious activities", "Barred numbers", "Users", "Settings"][array_rand(["A", "B", "C"])];
            $log['action'] = ["Consult", "Create", "Edit", "Delete", "Block"][array_rand(["Consult", "Create", "Edit", "Delete", "Block"])];

            $randomTimestamp = random_int($startDate->timestamp, $endDate->timestamp);
            $randomDateTime = date('Y-m-d H:i:s', $randomTimestamp);

            $log['date'] = $randomDateTime;
            $logs[] = $log;
        }
        usort($logs, function($a, $b) {
            return strtotime($b['date']) - strtotime($a['date']);
        });
        return view('webmaster.dashboard.dashboard')->with('subscriptions', $sortedSubscriptions)->with('logs', $logs);
    }
    public function settings()
    {
        return view('webmaster.settings.settings');
    }
}
