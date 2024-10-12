<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Customer extends Model
{
    use HasFactory;
    protected $fillable = ['name', 'birthday', 'birthday_location'];
    public function Phones()
    {
        return Subscription::where('customer', $this->id)->get();
    }
    public function Calls()
    {
        $total = 0;
        foreach($this->Phones() as $phone)
        {
            $total += $phone->Calls();
        }
        return $total;
    }
    public function SMS()
    {
        $total = 0;
        foreach($this->Phones() as $phone)
        {
            $total += $phone->SMS();
        }
        return $total;
    }
}
