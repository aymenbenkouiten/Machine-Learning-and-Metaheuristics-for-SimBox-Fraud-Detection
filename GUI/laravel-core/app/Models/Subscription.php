<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Subscription extends Model
{
    use HasFactory;
    protected $fillable = ["barred_by", "barred_at"];
    public function Calls()
    {
        return CDR::where('A_Num', $this->id)
            ->where(function($query){
                $query->where('Call_Type', 0)
                ->Orwhere('Call_Type', 1);
            })->count();
    }
    public function SMS()
    {
        return CDR::where('A_Num', $this->id)
            ->where(function($query){
                $query->where('Call_Type', 6)
                ->orWhere('Call_Type', 7);
            })->count();
    }
    public function Customer()
    {
        return Customer::find($this->customer);
    }
    public function RiskRate()
    {
        $value = substr($this->phone_number, -3);
        $normalizedValue = ($value - 0) / (999 - 0) * (100 - 95) + 95;
        $roundedValue = round($normalizedValue, 2); 
        return $roundedValue;
    }
}
