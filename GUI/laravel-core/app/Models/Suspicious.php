<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Suspicious extends Model
{
    use HasFactory;
    protected $fillable = ["checked_by", "response", "checked_at"];
    public function Subscription()
    {
        return Subscription::find($this->subscription);
    }
    
    public function Customer()
    {
        return Customer::find(Subscription::find($this->subscription)->customer);
    }
}
