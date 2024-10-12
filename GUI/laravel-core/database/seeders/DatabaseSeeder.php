<?php

namespace Database\Seeders;

use App\Models\CDR;
// use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use App\Models\User;
use App\Models\Customer;
use App\Models\Suspicious;
use App\Models\Subscription;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // User::factory(10)->create();

        $permissions = [];
        foreach(config('permissions') as $section=>$sectionPermission){
            foreach($sectionPermission as $permission){
                $permissions[] = $section.'_'.$permission;
            }
        }
        User::create([
            "name" => "Tarek NOUGHI", 
            "email" => "noughitarek@gmail.com", 
            "password" => Hash::make('password'),
            "role" => "admin",
            "permissions" => implode(',',$permissions),
        ]);
        User::create([
            "name" => "Aymen BENKOUITEN", 
            "email" => "noughitare2k@gmail.com", 
            "password" => Hash::make('password'),
            "role" => "admin",
            "permissions" => implode(',',$permissions),
        ]);
        User::create([
            "name" => "Ilyes KHENNAK", 
            "email" => "noughitare3k@gmail.com", 
            "password" => Hash::make('password'),
            "role" => "admin",
            "permissions" => implode(',',$permissions),
        ]);
        User::create([
            "name" => "Amine BENATMANE", 
            "email" => "noughitare4k@gmail.com", 
            "password" => Hash::make('password'),
            "role" => "admin",
            "permissions" => implode(',',$permissions),
        ]);

        $cdr = file_get_contents('laravel-core/database/seeds/cdr.csv');
        $rows = array_map('str_getcsv', explode("\n", $cdr));
        for ($i = 1; $i < 5000; $i++) {
            $row = $rows[$i];
            $data = [
                'Call_Type' => (int)$row[0],
                'Charging_Tm' => $row[1],
                'Call_Duration' => $row[2],
                'Telesrvc' => (int)$row[3],
                'LOCATION' => (int)$row[4],
                'A_Num' => (int)$row[5],
                'B_Num' => (int)$row[6],
                'cell_id' => (int)$row[7],
                'IMEI' => (int)$row[8],
                'TAC' => (int)$row[9],
                'DESTINATION_CAT' => $row[10],
                'MU_HANDSET_DUAL_SIM' => $row[11],
                'MU_Device_type_Segment' => $row[12],
                'MU_HANDSET_MOBILE_TECH' => $row[13],
            ];
            $sub = Subscription::find($data["A_Num"]);
            if(!$sub)
            {
                if (mt_rand(1, 10) == 1)
                {
                    $randomCustomer = Customer::inRandomOrder()->first();
                    if ($randomCustomer) {
                        $customer = $randomCustomer;
                    } else {
                        $customer = Customer::create([
                            'NIN' => $this->genererNumeroActeNaissance(),
                            'name' => $this->generateRandomAlgerianName(),
                            'birthday' => $this->generateRandomBirthday(),
                            'birthday_location' => $this->generateRandomBirthdayLocation(),
                        ]);
                    }
                }
                else
                {
                    $customer = Customer::create([
                        'NIN' => $this->genererNumeroActeNaissance(),
                        'name' => $this->generateRandomAlgerianName(),
                        'birthday' => $this->generateRandomBirthday(),
                        'birthday_location' => $this->generateRandomBirthdayLocation(),
                    ]);
                }
                $sub = Subscription::create([
                    'id' => $data["A_Num"],
                    'phone_number' => $this->generateRandomPhoneNumber(),
                    'established_by' => 'Espace Djezzy '.$this->generateRandomBirthdayLocation(),
                    'customer' => $customer->id
                ]);
            }
            CDR::create($data);
        }
        $subs = Subscription::inrandomorder()->take(50)->get();
        foreach ($subs as $sub) {
            Suspicious::create([
                'subscription' => $sub->id,
            ]);
        }
    }
    function genererNumeroActeNaissance() {
        $numero = "";
        $numero .= mt_rand(0, 1);
        $numero .= mt_rand(0, 1);
        $numero .= mt_rand(0, 1);
        $numero .= mt_rand(0, 1);
        $numero .= str_pad(mt_rand(0, 999), 3, "0", STR_PAD_LEFT);
        $numero .= str_pad(mt_rand(0, 9999), 4, "0", STR_PAD_LEFT);
        $numero .= str_pad(mt_rand(0, 9999), 4, "0", STR_PAD_LEFT);
        $numero .= str_pad(mt_rand(0, 99), 2, "0", STR_PAD_LEFT);
        $numero .= $this->calculerCleControle($numero);
        return $numero;
    }
    function calculerCleControle($numero) {
        $sum = 0;
        for ($i = 0; $i < strlen($numero); $i++) {
            $sum += (int)$numero[$i];
        }
        return $sum % 10;
    }
    function generateRandomAlgerianName() {
        $first_names = array(
            "Ahmed", "Mohammed", "Fatima", "Amina", "Youssef", "Laila", "Abdelkader", "Zahra", "Karim", "Sofia",
            "Ali", "Nora", "Omar", "Nadia", "Hassan", "Samia", "Malik", "Saida", "Hamza", "Amira",
            "Khaled", "Hafsa", "Rachid", "Houda", "Said", "Djamila", "Bilal", "Khadija", "Nabil", "Rim",
            "Mustapha", "Saliha", "Hicham", "Zineb", "Mounir", "Hayat", "Farid", "Wafa", "Reda", "Souad",
            "Mehdi", "Sana", "Amar", "Asma", "Nasser", "Aicha", "Tarek", "Nawal", "Salem", "Ines",
            "Rachida", "Fares", "Mouna", "Fouad", "Rania", "Walid", "Meryem", "Anis", "Soraya", "Abderrahmane",
            "Fadila", "Kamel", "Najat", "Brahim", "Sabrina", "Abdelhakim", "Amel", "Riyad", "Imane", "Azzedine",
            "Soumia", "Adel", "Latifa", "Yacine", "Houria", "Rachid", "Kenza", "Toufik", "Khadijah", "Mokhtar",
            "Nabila", "Walid", "Naima", "Lotfi", "Rabia", "Mohamed", "Salima", "Hocine", "Nadia", "Bachir",
            "Hadjer", "Amine", "Fatiha", "Lyes", "Noura", "Sofiane", "Warda", "Karim", "Nadjia", "Rachid",
            "Sara", "Yassine", "Nour", "Yasmina", "Rachid", "Halima", "Abdelaziz", "Dalila", "Yacine", "Fatima",
            "Nacer", "Salima", "Bilal", "Hayet", "Sofiane", "Zineb", "Ali", "Farida", "Kamel", "Houria",
            "Mohamed", "Djouher", "Hamid", "Salima", "Kamel", "Zohra", "Amir", "Soraya", "Said", "Hafida",
            "Karim", "Noura", "Mohamed", "Farida", "Khaled", "Karima", "Youssef", "Nour", "Kamel", "Nadia",
            "Nasreddine", "Naima", "Kamel", "Naima", "Mohamed", "Hadjira", "Abdelkader", "Sara", "Mohamed", "Salima",
            "Brahim", "Rahma", "Mohamed", "Yamina", "Ahmed", "Rim", "Amine", "Nawal", "Abdelkrim", "Kheira",
            "Mehdi", "Djamila", "Rachid", "Chafia", "Ali", "Khadija", "Omar", "Zohra", "Karim", "Kheira",
            "Nassim", "Djamila", "Houari", "Karima", "Khalid", "Hadjer", "Toufik", "Rahma", "Omar", "Fatiha",
            "Noureddine", "Nadia", "Amine", "Farida", "Abdelkader", "Houria", "Noureddine", "Nadia", "Mohamed", "Hafsa",
            "Hassan", "Nawal", "Houari", "Yasmina", "Ahmed", "Naima", "Younes", "Malika", "Hocine", "Yamina"
        );
    
        $last_names = array(
            "Benamar", "Bouzid", "Hamidi", "Ait Said", "Belkacem", "Slimani", "Khedira", "Toumi", "Bekkouche", "Haddad",
            "Mokhtar", "Bensaid", "Boumediene", "Lakhdari", "Rezki", "Azzouz", "Zerrouki", "Lakhdari", "Hammoudi", "Hadjouti",
            "Kaci", "Azzoun", "Lounis", "Hadjouti", "Ait Hamou", "Saci", "Boudjema", "Bouziane", "Ouafi", "Ouafi",
            "Ait Said", "Kamel", "Maiza", "Cherif", "Taleb", "Bouhaddi", "Benmohamed", "Belarbi", "Boulaouane", "Bekkouche",
            "Kherbouche", "Slimani", "Taleb", "Boumaza", "Ouaziz", "Benali", "Oudjani", "Benkhedda", "Boukerche", "Bouchenak",
            "Boukra", "Boudraa", "Benyahia", "Boumaza", "Boussad", "Bouneb", "Boudjellal", "Bougherara", "Boussad", "Bourekba",
            "Aissat", "Benkhaled", "Boukharouba", "Boudi", "Bouaziz", "Kouadria", "Boukhechba", "Kouadria", "Benarfa", "Bouaziz",
            "Boukhaloua", "Boudalia", "Boukerroucha", "Ouadahi", "Bouchenna", "Boukerroucha", "Boukara", "Benkara", "Boukhecha", "Bouchene",
            "Boukarroum", "Bouchene", "Boukhennoufa", "Bouchenak", "Boukerroum", "Boumaza", "Bouzidi", "Boudiaf", "Bouchemella", "Bouaziz",
            "Bouchara", "Bourezg", "Bouchireb", "Boukhechba", "Boukellal", "Boukhari", "Boukhenafar", "Bouhraoua", "Boukhechba", "Boukhenafar", "Bouhafs", "Bouarroudj", "Boukrine", "Bouchouicha", "Boukhari",
            "Bouhafs", "Boukherouaa", "Bouarroudj", "Bouzghiba", "Boukhechba", "Boukherouaa", "Boussouar", "Boukaouar", "Boukhersa", "Bouanani",
            "Boukhedenna", "Boukhersa", "Boudinani", "Boukhersa", "Boukhemenna", "Bouziane", "Boukhersa", "Boukhoubza", "Bouhezza", "Boudhane",
            "Bouaoune", "Bouhafs", "Boukhemenna", "Bouhemadou", "Boudissa", "Bouzghiba", "Boudiaf", "Boukhemaissa", "Bouheroua", "Boukhemessa",
            "Boukerrou", "Boumessaoud", "Boukhemissa", "Bouhemissa", "Bouhafs", "Boumessaoud", "Bouzar", "Boumessaoud", "Bouzghiba", "Bouaoune",
            "Bouzghiba", "Bouhafs", "Bouzar", "Bouhezza", "Boukherouaa", "Bouhafs", "Bouzghiba", "Bouziane", "Boukhemessa", "Boukhemessa",
            "Bouzghiba", "Bouziane", "Boukherouaa", "Bouziane", "Boukemissa", "Bouzghiba", "Boukhezza", "Bouziane", "Boukhari", "Bouziane",
            "Boukhemessa", "Bouzar", "Bouhezza", "Boukhemessa", "Boumessaoud", "Boukhari", "Boukemissa", "Bouzar", "Boumessaoud", "Bouhemissa"
        );
    
        $random_first_name = $first_names[array_rand($first_names)];
        $random_last_name = $last_names[array_rand($last_names)];
    
        return $random_first_name . ' ' . $random_last_name;
    }
    function generateRandomBirthday() {
        $start_date = strtotime("1920-01-01");
        $end_date = strtotime("2004-12-31");
    
        $random_timestamp = mt_rand($start_date, $end_date);
    
        return date("d/m/Y", $random_timestamp);
    }
    function generateRandomBirthdayLocation() {
        $locations = array(
            "Algiers", "Oran", "Constantine", "Annaba", "Blida", "Batna", "Djelfa", "Sétif", "Sidi Bel Abbès", "Biskra",
            "Tébessa", "El Oued", "Skikda", "Tiaret", "Béjaïa", "Tlemcen", "Ouargla", "Béchar", "Mostaganem", "Tizi Ouzou",
            "Bordj Bou Arréridj", "El Achir", "Médéa", "Tiarat", "Aïn Beïda", "Relizane", "Jijel", "Guelma", "Aïn Oussera",
            "Khenchela", "Sougueur", "Khemis Miliana", "El Bayadh", "Aflou", "M'Sila", "Tindouf", "Reggane", "El Tarf", "Aïn El Hammam",
            "Boghni", "Sour El Ghozlane", "Ksar Chellala", "Tolga", "Barika", "Ras El Oued", "Frenda", "Aïn Touta", "El Attaf",
            "Oued El Abtal", "Tébessa", "Touggourt", "Laghouat", "Aïn Taya", "Aïn Defla", "Hammam Bou Hadjar", "Aïn Arnat", "Dar El Beïda",
            "Bir El Djir", "Dar Chioukh", "Saïda", "Mouzaïa", "El Affroun", "Sidi Aïssa", "Besbes", "Berrahal", "Sidi Abdelli",
            "Sidi Khaled", "Hennaya", "Boumahra Ahmed", "Bir El Ater", "Aïn Bessem", "El Kala", "Beni Saf", "Aïn Benian", "Hennaya",
            "El Attaf", "Thenia", "Sidi M'Hamed", "Larbaâ", "Sidi M'Hamed", "Berrouaghia", "Aïn El Turk", "Rouissat", "Tirmitine",
            "Beni Amrane", "El Hachem", "Tolga", "Hadjout", "Mila", "Bouïra", "Chlef", "Adrar", "Naâma", "M'Sila",
            "Oum El Bouaghi", "Mascara", "El Bayadh", "Bouïra", "Bouïra", "Aïn Taya", "Berriane", "Berriane", "Berriane",
            "Tamanrasset", "M'Sila", "Djanet", "Ouled Djellal", "Djamaa", "Aflou", "Bordj Badji Mokhtar", "Bordj Omar Driss");
        return $locations[array_rand($locations)];
    }
    function generateRandomPhoneNumber() {
        $prefix = '0';
        $middleNumbers = 7;
        $suffix = mt_rand(70000000, 99999999);
    
        return $prefix . $middleNumbers . $suffix;
    }
}
