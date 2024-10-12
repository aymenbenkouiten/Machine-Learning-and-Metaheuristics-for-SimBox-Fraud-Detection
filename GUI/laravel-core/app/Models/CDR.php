<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class CDR extends Model
{
    use HasFactory;
    public function Call_Type()
    {
        if($this->Call_Type == 1 && $this->Telesrvc == 11)
            return "CALL-IN";
        if($this->Call_Type == 0 && $this->Telesrvc == 11)
            return "CALL-OUT";
        if($this->Call_Type == 7)
            return "SMS-IN";
        if($this->Call_Type == 6 && $this->Telesrvc == 22)
            return "SMS-OUT";
        if($this->Call_Type == 100)
            return "FORWARD";
        if($this->Call_Type == 2)
            return "ROAMING FORWARD";
        if($this->Call_Type == 12)
            return "EMERGENCY-CALL";
        return "UNKNOWN";
    }
    
    public function Customer()
    {
        return Customer::find(Subscription::find($this->A_Num)->customer);
    }
    public function B_Num()
    {
        $sub = Subscription::find($this->B_Num);
        if(!$sub)
        {
            return new Customer([
                'name' => "n/a",
                'birthday' => "n/a",
                'birthday_location' => "n/a",
            ]);
        }
        return Customer::find($sub->customer);
    }
    public function generateIMEI()
    {
        $imei = '';
        for ($i = 0; $i < 14; $i++) {
            $imei .= mt_rand(0, 9);
        }
        $imeiArray = str_split($imei);
        $sum = 0;
        for ($i = 0; $i < 14; $i++) {
            if ($i % 2 == 0) {
                $sum += $imeiArray[$i];
            } else {
                $double = $imeiArray[$i] * 2;
                $sum += array_sum(str_split($double));
            }
        }
        $checkDigit = (10 - ($sum % 10)) % 10;

        return $imei . $checkDigit;
    }
    public function generateTAC()
    {
        $phoneModels = ['iPhone', 'Samsung Galaxy', 'Google Pixel', 'Huawei Mate', 'OnePlus', 'Xiaomi Mi', 'LG G', 'Sony Xperia'];
        return $phoneModels[array_rand($phoneModels)];
    }
}
